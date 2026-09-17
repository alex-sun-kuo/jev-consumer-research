import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fake_jev import fake_build_opener
from recipes.persona_adapter import adapt_personas, read_personas, with_personas
from recipes.run_study import build_state
from recipes.summarize import available_groups

ROOT = Path(__file__).resolve().parents[1]


class PersonaAdapterTests(unittest.TestCase):
    def setUp(self):
        self.mapping = json.loads((ROOT / 'data' / 'persona_mapping.json').read_text())
        self.records = read_personas(ROOT / 'data' / 'example_personas.csv')
        self.study = json.loads((ROOT / 'data' / 'study.json').read_text())

    def test_csv_custom_fields_and_unknowns(self):
        self.records[0]['email'] = 'not-selected@example.test'
        profiles = adapt_personas(self.records, self.mapping)
        self.assertIsNone(profiles[0]['demographics']['ethnicity'])
        self.assertEqual(profiles[0]['context']['budget_usd'], 35.0)
        self.assertEqual(profiles[0]['context']['shopping_style'], 'Reads return terms')
        state = build_state(profiles[0], self.study['stimuli'][0])
        self.assertIn('shopping_style', state['profile']['context'])
        self.assertNotIn('email', json.dumps(state))
        self.assertNotIn('age_band', profiles[0]['demographics'])
        self.assertIn('context.shopping_style', available_groups(profiles))

    def test_nested_json_and_explicit_types(self):
        mapping = {'id_field': 'meta.id', 'demographics': {'custom_flag': 'survey.flag'},
                   'context': {'interests': 'survey.interests', 'quantity': 'survey.count'},
                   'types': {'demographics.custom_flag': 'bool', 'context.quantity': 'int'}}
        rows = [{'meta': {'id': 1}, 'survey': {'flag': 'false', 'count': '2', 'interests': ['repair', 'reuse']}}]
        profile = adapt_personas(rows, mapping)[0]
        self.assertEqual(profile['id'], '1')
        self.assertIs(profile['demographics']['custom_flag'], False)
        self.assertEqual(profile['context']['quantity'], 2)
        self.assertEqual(profile['context']['interests'], ['repair', 'reuse'])
        rows[0]['survey']['interests'].append('changed')
        self.assertEqual(profile['context']['interests'], ['repair', 'reuse'])

    def test_json_and_jsonl_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            for extension in ('json', 'jsonl'):
                path = Path(directory) / f'people.{extension}'
                path.write_text(json.dumps(self.records) if extension == 'json' else
                                '\n'.join(json.dumps(r) for r in self.records))
                self.assertEqual(read_personas(path), self.records)

    def test_bad_ids_mapping_types_and_missing_budget_fail(self):
        duplicate = self.records + [self.records[0]]
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            adapt_personas(duplicate, self.mapping)
        for bad in ('not-a-number', 'nan', True, ''):
            rows = copy.deepcopy(self.records)
            rows[0]['product_budget'] = bad
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                profiles = adapt_personas(rows, self.mapping)
                with_personas(self.study, profiles, revision='import-v1')
        mapping = copy.deepcopy(self.mapping)
        mapping['context']['need'] = 'typo_column'
        with self.assertRaisesRegex(ValueError, 'Source field not found'):
            adapt_personas(self.records, mapping)
        mapping = copy.deepcopy(self.mapping)
        mapping['types']['unknown.target'] = 'float'
        with self.assertRaises(ValueError):
            adapt_personas(self.records, mapping)

    def test_panel_replacement_preserves_stimuli_and_source(self):
        before = copy.deepcopy(self.study)
        profiles = adapt_personas(self.records, self.mapping)
        changed = with_personas(self.study, profiles, revision='import-v1')
        self.assertEqual(changed['stimuli'], before['stimuli'])
        changed['profiles'][0]['context']['need'] = 'changed'
        self.assertEqual(self.study, before)
        self.assertNotEqual(profiles[0]['context']['need'], 'changed')
        with self.assertRaises(ValueError):
            with_personas(self.study, profiles, revision=self.study['revision'])

    def test_notebook_runs_with_imported_panel_missing_default_demographics(self):
        nb = json.loads((ROOT / 'cookbooks' / 'consumer_focus_groups.ipynb').read_text())
        original = Path.cwd()
        try:
            with tempfile.TemporaryDirectory() as directory:
                os.chdir(directory)
                scope = {'__name__': '__main__'}
                with contextlib.redirect_stdout(io.StringIO()), patch.dict(
                        os.environ, {'TYPESAFE_API_KEY': 'test-only'}), patch(
                        'urllib.request.build_opener', fake_build_opener), patch('time.sleep'):
                    for cell in nb['cells']:
                        if cell['cell_type'] != 'code':
                            continue
                        source = ''.join(cell['source'])
                        if cell['id'] == 'persona-import-code':
                            source = source.replace('PERSONA_FILE = None', 'PERSONA_FILE = ' + repr(str(ROOT / 'data' / 'example_personas.csv')))
                        exec(compile(source, cell['id'], 'exec'), scope)
                self.assertEqual(len(scope['rows']), 30)
                self.assertEqual(len(scope['masked_rows']), 30)
                self.assertTrue(all(row['status'] == 'ok' for row in scope['rows']))
                self.assertNotIn('age_band', scope['group_fields'])
                self.assertIn('context.shopping_style', scope['group_fields'])
                self.assertTrue((scope['run_dir'] / 'charts.png').exists())
        finally:
            os.chdir(original)


if __name__ == '__main__':
    unittest.main()
