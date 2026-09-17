import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fake_jev import fake_build_opener, study_fixture
from recipes.run_study import build_state, main, questions, validate_study
from recipes.summarize import comparisons, diagnostic_report, group_values, load

ROOT = Path(__file__).resolve().parents[1]


class RichProfileTests(unittest.TestCase):
    def setUp(self):
        self.study = json.loads((ROOT / 'data' / 'study.json').read_text())
        self.profile = self.study['profiles'][0]
        self.stimulus = self.study['stimuli'][0]

    def test_projection_passes_rich_fields_and_masks_entire_block(self):
        visible = build_state(self.profile, self.stimulus)
        masked = build_state(self.profile, self.stimulus, mask_demographics=True)
        self.assertEqual(visible['profile']['demographics'], self.profile['demographics'])
        self.assertEqual(masked['profile'], {'context': self.profile['context']})
        self.assertEqual(visible['stimulus'], masked['stimulus'])
        self.assertNotIn('id', visible['profile'])
        self.assertNotIn('provenance', visible['profile'])
        self.assertNotIn('variant', visible['stimulus'])
        self.assertNotIn('id', visible['stimulus'])
        self.assertIn('proof_required', visible['profile']['context'])

    def test_missing_or_invalid_budget_is_not_replaced_by_income(self):
        for value in (None, True, -1, float('nan'), float('inf'), '35'):
            with self.subTest(value=value):
                changed = copy.deepcopy(self.study)
                changed['profiles'][0]['context']['budget_usd'] = value
                with self.assertRaisesRegex(ValueError, 'product_budget'):
                    validate_study(changed)

    def test_optional_unknown_demographic_fields_remain_optional(self):
        self.profile['demographics'] = {'ethnicity': None}
        validate_study(self.study)
        self.assertEqual(group_values([self.profile], 'ethnicity'), ['unspecified'])
        self.assertEqual(group_values([self.profile, {'demographics': {}}], 'ethnicity'),
                         ['unspecified', 'unspecified'])
        self.assertEqual(group_values(self.study['profiles'], 'household_size')[0], 'unspecified')

    def test_context_groups_and_misspelled_or_list_fields(self):
        self.assertEqual(group_values(self.study['profiles'], 'context.purchase_stage'),
                         ['actively comparing', 'browsing', 'browsing', 'actively comparing'])
        for field in ('incom_level', 'context.unknown', 'context.decision_criteria', 'provenance.source'):
            with self.subTest(field=field), self.assertRaises(ValueError):
                group_values(self.study['profiles'], field)

    def test_unavailable_diagnostics_do_not_become_zero(self):
        qs = questions(self.stimulus)
        row = dict(profile_id='p1', stimulus_id='ad_a', kind='ad', variant='A',
                   context=self.profile['context'], demographics=self.profile['demographics'],
                   status='ok', answers=study_fixture(build_state(self.profile, self.stimulus), qs)['answers'])
        unavailable = dict(row, profile_id='p2', status='unavailable', answers=None)
        report = diagnostic_report([row, unavailable])
        self.assertIn('| 1 | 1 | 0.450 | 0.990 | 0.800 | 0.100 |', report)
        self.assertIn('| 0 | 1 | unavailable |', diagnostic_report([unavailable]))

    def test_matched_pair_cannot_change_segment(self):
        a = dict(profile_id='p1', kind='ad', variant='A', demographics={'income_level': 'low'})
        b = dict(a, variant='B', demographics={'income_level': 'high'})
        with self.assertRaisesRegex(ValueError, 'inconsistent_pair_group'):
            comparisons([a, b], 'income_level')

    def test_notebook_runs_standalone_and_matches_cli(self):
        notebook = json.loads((ROOT / 'cookbooks' / 'consumer_focus_groups.ipynb').read_text())
        code_cells = []
        for index, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                self.assertEqual(notebook['cells'][index - 1]['cell_type'], 'markdown')
                code_cells.append(compile(''.join(cell['source']), f'notebook-cell-{index}', 'exec'))
        original_cwd = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as directory:
                os.chdir(directory)
                scope = {'__name__': '__main__'}
                with contextlib.redirect_stdout(io.StringIO()), patch.dict(
                    os.environ, {'TYPESAFE_API_KEY': 'test-only'}
                ), patch('urllib.request.build_opener', fake_build_opener), patch('time.sleep'):
                    for cell in code_cells:
                        exec(cell, scope)
                    self.assertEqual(scope['study'], self.study)
                    self.assertEqual(len(scope['rows']), 24)
                    self.assertEqual(len(scope['masked_rows']), 24)
                    self.assertTrue(all(row['status'] == 'ok' for row in scope['rows'] + scope['masked_rows']))
                    self.assertEqual(scope['sensitivity_result']['mean_absolute_score_change'], 0)
                    self.assertAlmostEqual(scope['validation_result']['brier'], 0.18375)
                    self.assertEqual(len([p for p in scope['run_dir'].rglob('*') if p.is_file()]), 14)
                    self.assertTrue((scope['run_dir'] / 'charts.png').read_bytes().startswith(b'\x89PNG'))
                    self.assertIn('<svg', (scope['run_dir'] / 'charts.svg').read_text())
                    self.assertIn('context', scope['rows'][0])
                    self.assertIn('profile_provenance', scope['rows'][0])
                    self.assertIn('unspecified', scope['report'](scope['rows'], 'ethnicity'))
                    for masked, name in ((False, 'rows'), (True, 'masked_rows')):
                        output = Path(directory) / f'cli-{masked}.jsonl'
                        argv = ['run_study.py', '--study', str(ROOT / 'data' / 'study.json'), '--output', str(output)]
                        if masked:
                            argv.append('--mask-demographics')
                        with patch('sys.argv', argv):
                            main()
                        self.assertEqual(load(output), scope[name])
                    # Run All again in the same kernel must create a new output folder.
                    first_run = scope['run_dir']
                    for cell in code_cells:
                        exec(cell, scope)
                    self.assertNotEqual(first_run, scope['run_dir'])
                    self.assertTrue(first_run.exists())
        finally:
            os.chdir(original_cwd)


if __name__ == '__main__':
    unittest.main()
