import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib.pyplot as plt

from fake_jev import fake_build_opener, study_fixture
from recipes.concept_read import concept_report, stimulus_read
from recipes.run_study import build_state, questions
from recipes.visualize_results import plot_concept

ROOT = Path(__file__).resolve().parents[1]


class ConceptReadTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def setUp(self):
        study = json.loads((ROOT / 'data' / 'study.json').read_text())
        self.rows = [dict(
            study_id='test', revision='1', mode='live_simulation', demographics_masked=False,
            profile_id=profile['id'], stimulus_id=stimulus['id'], kind=stimulus['kind'],
            variant=stimulus['variant'], status='ok',
            answers=study_fixture(build_state(profile, stimulus), questions(stimulus))['answers'],
        ) for profile in study['profiles'] for stimulus in study['stimuli'] if stimulus['variant'] == 'A']

    def test_reads_average_each_stimulus_separately(self):
        reads = stimulus_read(self.rows)
        self.assertEqual([(r['kind'], r['stimulus_id']) for r in reads],
                         [('ad', 'ad_a'), ('product', 'product_a'), ('website', 'website_a')])
        ad = reads[0]
        self.assertEqual((ad['available'], ad['unavailable']), (4, 0))
        self.assertAlmostEqual(ad['actions']['engage'], 0.23)
        self.assertAlmostEqual(sum(ad['actions'].values()), 1)
        self.assertAlmostEqual(ad['barriers']['price_objection'], 0.5)
        self.assertEqual([profile for profile, _, _ in ad['profiles']], ['p1', 'p2', 'p3', 'p4'])

    def test_unavailable_rows_do_not_become_zero(self):
        rows = copy.deepcopy(self.rows)
        rows[0].update(status='unavailable', answers=None)
        ad = next(r for r in stimulus_read(rows) if r['stimulus_id'] == 'ad_a')
        self.assertEqual((ad['available'], ad['unavailable']), (3, 1))
        self.assertNotIn(rows[0]['profile_id'], [profile for profile, _, _ in ad['profiles']])
        empty = stimulus_read([dict(r, status='unavailable', answers=None)
                               for r in self.rows if r['stimulus_id'] == 'ad_a'])[0]
        self.assertEqual(empty['available'], 0)
        self.assertIsNone(empty['actions'])
        report = concept_report([empty])
        self.assertIn('Unavailable: 4', report)
        self.assertNotIn('0.000', report)

    def test_invalid_probabilities_rejected(self):
        for mutate in (lambda r: r['answers']['action']['probabilities'].update(engage=1.5),
                       lambda r: r['answers']['relevant'].update(noul=float('nan'))):
            rows = copy.deepcopy(self.rows)
            mutate(rows[0])
            with self.assertRaisesRegex(ValueError, 'invalid_probability'):
                stimulus_read(rows)

    def test_report_lists_profiles_with_divergence(self):
        report = concept_report(stimulus_read(self.rows))
        self.assertIn('## ad / ad_a', report)
        self.assertIn('| price objection | 0.500 |', report)
        self.assertIn('- p1: engage 0.180 (-0.050), sentiment mixed', report)

    def test_plot_concept_bars_match_scores(self):
        read = stimulus_read(self.rows)[0]
        figure = plot_concept(read)
        panels = (list(read['actions'].items()), [(p, e) for p, e, _ in read['profiles']])
        for axis, values in zip(figure.axes, panels):
            bars = axis.containers[0]
            self.assertEqual(len(bars), len(values))
            for bar, (_, value) in zip(bars, values):
                self.assertAlmostEqual(bar.get_width(), value)
        empty = plot_concept(dict(read, available=0, actions=None, profiles=[]))
        self.assertTrue(any('No available results' in text.get_text() for text in empty.axes[0].texts))

    def test_notebook_runs_standalone_and_matches_module(self):
        notebook = json.loads((ROOT / 'cookbooks' / 'concept_read.ipynb').read_text())
        code_cells = []
        for index, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                self.assertEqual(notebook['cells'][index - 1]['cell_type'], 'markdown')
                code_cells.append(compile(''.join(cell['source']), f'concept-cell-{index}', 'exec'))
        original = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as directory:
                os.chdir(directory)
                scope = {'__name__': '__main__'}
                with contextlib.redirect_stdout(io.StringIO()), patch.dict(
                    os.environ, {'TYPESAFE_API_KEY': 'test-only'}
                ), patch('urllib.request.build_opener', fake_build_opener), patch('time.sleep'):
                    for cell in code_cells:
                        exec(cell, scope)
                self.assertEqual(len(scope['rows']), 4)
                self.assertTrue(all(row['status'] == 'ok' for row in scope['rows']))
                self.assertEqual(scope['concept_reads'], stimulus_read(scope['rows']))
                self.assertEqual(scope['concept_brief'], concept_report(scope['concept_reads']))
                engages = [engage for _, engage, _ in scope['concept_reads'][0]['profiles']]
                self.assertEqual(engages, [0.18, 0.5, 0.5, 0.18])
                self.assertTrue((scope['run_dir'] / 'concept_read.md').exists())
                self.assertTrue((scope['run_dir'] / 'concept.png').read_bytes().startswith(b'\x89PNG'))
        finally:
            os.chdir(original)


if __name__ == '__main__':
    unittest.main()
