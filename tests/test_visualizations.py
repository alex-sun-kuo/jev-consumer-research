import copy
import json
from pathlib import Path
import unittest

import matplotlib.pyplot as plt

from fake_jev import study_fixture
from recipes.run_study import build_state, questions
from recipes.visualize_results import chart_data, plot_charts

ROOT = Path(__file__).resolve().parents[1]


class VisualizationTests(unittest.TestCase):
    def tearDown(self):
        plt.close('all')

    def setUp(self):
        study = json.loads((ROOT / 'data' / 'study.json').read_text())
        self.rows = []
        for profile in study['profiles']:
            for stimulus in study['stimuli']:
                self.rows.append(dict(
                    profile_id=profile['id'], kind=stimulus['kind'], variant=stimulus['variant'],
                    mode='live_simulation', status='ok',
                    answers=study_fixture(build_state(profile, stimulus), questions(stimulus))['answers'],
                ))

    def test_pmf_uses_one_surface_and_preserves_unknown(self):
        result = chart_data(self.rows)
        self.assertEqual(result['complete_pairs'], 4)
        self.assertEqual(result['excluded_pairs'], 0)
        actions = {label: (a, b) for label, a, b in result['actions']}
        self.assertAlmostEqual(actions['engage'][0], 0.23)
        self.assertAlmostEqual(actions['engage'][1], 0.34)
        self.assertEqual(actions['unknown'], (0.1, 0.1))
        for variant in (1, 2):
            self.assertAlmostEqual(sum(row[variant] for row in result['actions']), 1)
        deltas = [round(b-a, 2) for _, a, b in result['profiles']]
        self.assertEqual(deltas, [0, 0.22, 0.22, 0])

    def test_unavailable_pair_excluded_from_both_variants(self):
        next(row for row in self.rows if row['profile_id'] == 'p2' and row['kind'] == 'ad'
             and row['variant'] == 'B').update(status='unavailable', answers=None)
        result = chart_data(self.rows)
        self.assertEqual(result['complete_pairs'], 3)
        self.assertEqual(result['excluded_pairs'], 1)
        actions = {label: (a, b) for label, a, b in result['actions']}
        self.assertAlmostEqual(actions['engage'][0], (0.18+0.28+0.18)/3)
        self.assertNotIn('p2', [profile for profile, _, _ in result['profiles']])

    def test_empty_complete_pairs_are_not_plotted_as_zero(self):
        rows = [row for row in self.rows if row['variant'] == 'A']
        data = chart_data(rows)
        self.assertEqual(data['complete_pairs'], 0)
        self.assertEqual(data['excluded_pairs'], 4)
        figure = plot_charts(data)
        self.assertTrue(any('No complete A/B pairs' in text.get_text() for text in figure.axes[0].texts))
        self.assertTrue(all(not axis.patches for axis in figure.axes))

    def test_invalid_distributions_and_duplicate_variants_rejected(self):
        for bad in (float('nan'), -0.1, 0.9):
            rows = copy.deepcopy(self.rows)
            rows[0]['answers']['action']['probabilities']['unknown'] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                chart_data(rows)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            chart_data(self.rows + [self.rows[0]])

    def test_plot_preserves_scores_without_confidence_intervals(self):
        data = chart_data(self.rows)
        figure = plot_charts(data)
        for axis, values in zip(figure.axes, (data['actions'], data['profiles'])):
            for index, bars in enumerate(axis.containers):
                self.assertEqual(len(bars), len(values))
                for bar, value in zip(bars, values):
                    self.assertAlmostEqual(bar.get_width(), value[index+1])
            self.assertFalse(axis.lines, 'No confidence interval lines should be drawn')
            self.assertEqual(axis.get_xlim()[0], 0)

    def test_profile_labels_are_plain_text(self):
        data = chart_data(self.rows)
        data['profiles'][0] = ('$profile_1$', 0.18, 0.18)
        figure = plot_charts(data)
        label = figure.axes[1].get_yticklabels()[0]
        self.assertIn('$profile_1$', label.get_text())
        self.assertIn('B−A +0.000', label.get_text())
        self.assertFalse(label.get_parse_math())


if __name__ == '__main__':
    unittest.main()
