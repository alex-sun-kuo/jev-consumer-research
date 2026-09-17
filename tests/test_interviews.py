import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from recipes.analyze_interviews import (analyze_interviews, interview_questions, interview_report,
                                interview_state, parse_interview_text, selected_evidence, validate_interviews)
from fake_jev import fake_build_opener, interview_fixture
from recipes.run_study import validate_answers

ROOT = Path(__file__).resolve().parents[1]


class InterviewTests(unittest.TestCase):
    def setUp(self):
        self.study = json.loads((ROOT / 'data' / 'interviews_example.json').read_text())
        self.state = interview_state(self.study, self.study['interviews'][0], 'p1')
        self.questions = interview_questions(self.state)

    def test_only_target_participant_quotes_are_eligible(self):
        for key, question in self.questions.items():
            if key.startswith('evidence_'):
                self.assertEqual(set(question['criteria']), {'t2', 't4', 'none'})
        body = interview_fixture(self.state, self.questions)
        answers = validate_answers(body, self.questions)
        evidence = selected_evidence(self.state, answers)
        self.assertEqual(evidence['main_barrier']['quote'], self.study['interviews'][0]['turns'][1]['text'])
        answers['evidence_main_barrier']['choice'] = 't1'
        with self.assertRaises(ValueError):
            selected_evidence(self.state, answers)

    def test_speaker_labeled_text_and_unknown_speakers(self):
        interview = parse_interview_text('moderator: Why?\np1: I need proof.\nEspecially a leak test.',
                                        interview_id='one', participant='p1')
        self.assertEqual(interview['turns'][1]['text'], 'I need proof.\nEspecially a leak test.')
        self.assertEqual(interview['turns'][1]['role'], 'participant')
        with self.assertRaisesRegex(ValueError, 'Unknown speaker'):
            parse_interview_text('p2: hello', interview_id='one', participant='p1')

    def test_duplicate_ids_roles_and_empty_sessions_rejected(self):
        for mutate in (
            lambda s: s['interviews'][0]['turns'].append(s['interviews'][0]['turns'][0]),
            lambda s: s['interviews'][0]['turns'][3].update(role='moderator'),
            lambda s: s['interviews'][0]['turns'][3].update(speaker='p2'),
            lambda s: s.update(interviews=[]),
            lambda s: s['interviews'][0].update(turns=[]),
        ):
            changed = copy.deepcopy(self.study)
            mutate(changed)
            with self.assertRaises(ValueError):
                validate_interviews(changed)

    def test_end_to_end_and_recommendations(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'TYPESAFE_API_KEY': 'test-only'}), patch(
                'urllib.request.build_opener', fake_build_opener), patch('recipes.analyze_interviews.time.sleep'):
            output = Path(directory) / 'results.jsonl'
            rows = analyze_interviews(self.study, output)
            self.assertEqual(len(rows), 3)
            self.assertTrue(all(row['status'] == 'ok' for row in rows))
            self.assertEqual([r['answers']['research_next_step']['choice'] for r in rows],
                             ['test_price_value', 'test_proof', 'investigate_fit'])
            self.assertEqual(rows[2]['answers']['behavior_basis']['choice'], 'both')
            self.assertEqual(len(output.read_text().splitlines()), 3)
            with self.assertRaises(FileExistsError):
                analyze_interviews(self.study, output)
            rows[0]['evidence']['research_next_step'] = None
            report = interview_report(rows[:1])
            self.assertIn('Ask a neutral follow-up', report)
            self.assertIn('withheld', report)
            self.assertNotIn('Test price/value framing', report)

    def test_limits_before_network_and_http_stop_keeps_partial_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'results.jsonl'
            with self.assertRaises(ValueError):
                analyze_interviews(self.study, output, max_calls=2)
            self.assertFalse(output.exists())
            with patch.dict(os.environ, {'TYPESAFE_API_KEY': 'test-only'}), patch(
                'recipes.analyze_interviews.jev', side_effect=RuntimeError('http_429')
            ) as provider:
                with self.assertRaisesRegex(RuntimeError, 'Stopped'):
                    analyze_interviews(self.study, output)
            provider.assert_called_once()
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['status'], 'unavailable')

    def test_live_path_uses_typed_questions_and_resolves_evidence(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {'TYPESAFE_API_KEY': 'test-only'}), patch(
            'recipes.analyze_interviews.jev', side_effect=lambda state, qs, model: interview_fixture(state, qs)
        ) as provider, patch('recipes.analyze_interviews.time.sleep'):
            rows = analyze_interviews(self.study, Path(directory) / 'live.jsonl')
        self.assertEqual(provider.call_count, 3)
        self.assertTrue(all(row['mode'] == 'live_interview_analysis' for row in rows))
        self.assertTrue(all(row['status'] == 'ok' for row in rows))
        self.assertEqual(len(provider.call_args.args[1]), 10)

    def test_notebook_runs_standalone_twice_and_matches_module(self):
        notebook = json.loads((ROOT / 'cookbooks' / 'interview_analysis.ipynb').read_text())
        original = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as directory:
                os.chdir(directory)
                scope = {'__name__': '__main__'}
                with contextlib.redirect_stdout(io.StringIO()), patch.dict(
                        os.environ, {'TYPESAFE_API_KEY': 'test-only'}), patch(
                        'urllib.request.build_opener', fake_build_opener), patch('time.sleep'):
                    for _ in range(2):
                        for index, cell in enumerate(notebook['cells']):
                            if cell['cell_type'] == 'code':
                                self.assertEqual(notebook['cells'][index-1]['cell_type'], 'markdown')
                                exec(compile(''.join(cell['source']), cell['id'], 'exec'), scope)
                    expected = analyze_interviews(self.study, Path(directory) / 'module.jsonl')
                self.assertEqual(scope['interview_results'], expected)
                self.assertEqual(scope['interview_study'], self.study)
                self.assertTrue((scope['interview_run_dir'] / 'research_brief.md').exists())
                self.assertEqual(len(list((Path(directory) / 'interview_runs').iterdir())), 2)
        finally:
            os.chdir(original)


if __name__ == '__main__':
    unittest.main()
