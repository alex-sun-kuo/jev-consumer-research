import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import urllib.error

from fake_jev import study_fixture
from recipes.run_study import questions, validate_answers, validate_study, jev, NoRedirect
from recipes.summarize import comparisons, load, sensitivity
from recipes.validate_behavior import evaluate

ROOT = Path(__file__).resolve().parents[1]


class Tests(unittest.TestCase):
    def setUp(self):
        self.study = json.loads((ROOT / "data" / "study.json").read_text())
        self.state = {"profile": self.study["profiles"][0], "stimulus": self.study["stimuli"][0]}
        self.qs = questions(self.state["stimulus"])
        self.body = study_fixture(self.state, self.qs)

    def row(self, variant="A", status="ok"):
        return {"study_id": "test", "revision": "1", "mode": "live_simulation",
                "profile_id": "p", "stimulus_id": variant, "kind": "ad", "variant": variant,
                "demographics": {"age_band": "25-34"}, "demographics_masked": False,
                "status": status, "answers": copy.deepcopy(self.body["answers"])}

    def test_study(self):
        validate_study(self.study)

    def test_duplicate_profile(self):
        self.study["profiles"].append(self.study["profiles"][0])
        with self.assertRaises(ValueError):
            validate_study(self.study)

    def test_fake_jev_bodies_pass_validation(self):
        for p in self.study["profiles"]:
            for s in self.study["stimuli"]:
                q = questions(s)
                validate_answers(study_fixture({"profile": p, "stimulus": s}, q), q)

    def test_invalid_probabilities(self):
        for bad in (True, -1, 2, "0.5", float("nan")):
            body = copy.deepcopy(self.body)
            body["answers"]["relevant"]["noul"] = bad
            with self.assertRaises(ValueError):
                validate_answers(body, self.qs)

    def test_missing_answer(self):
        del self.body["answers"]["action"]
        with self.assertRaises(ValueError):
            validate_answers(self.body, self.qs)

    def test_invalid_sum(self):
        self.body["answers"]["action"]["probabilities"]["ignore"] = 0.9
        with self.assertRaises(ValueError):
            validate_answers(self.body, self.qs)

    def test_wrong_choice(self):
        self.body["answers"]["action"]["choice"] = "unknown"
        with self.assertRaises(ValueError):
            validate_answers(self.body, self.qs)

    def test_missing_category(self):
        del self.body["answers"]["action"]["probabilities"]["unknown"]
        with self.assertRaises(ValueError):
            validate_answers(self.body, self.qs)

    def test_pairs(self):
        value = next(iter(comparisons([self.row(), self.row("B")]).values()))
        self.assertEqual(value["deltas"], [0])

    def test_unavailable_not_zero(self):
        value = next(iter(comparisons([self.row(), self.row("B", "unavailable")]).values()))
        self.assertEqual(value["deltas"], [])
        self.assertEqual(value["unavailable_pairs"], 1)

    def test_missing_pair(self):
        self.assertEqual(next(iter(comparisons([self.row()]).values()))["unavailable_pairs"], 1)

    def test_sensitivity(self):
        a, b = self.row(), self.row()
        b["demographics_masked"] = True
        self.assertEqual(sensitivity([a], [b])["mean_absolute_score_change"], 0)

    def test_sensitivity_unmatched(self):
        a, b = self.row(), self.row("B")
        b["demographics_masked"] = True
        with self.assertRaises(ValueError):
            sensitivity([a], [b])

    def test_duplicate_output(self):
        with tempfile.TemporaryDirectory() as d:
            f = Path(d) / "r.jsonl"
            f.write_text((json.dumps(self.row()) + "\n") * 2)
            with self.assertRaises(ValueError):
                load(f)

    def test_brier(self):
        data = {"provenance": "test", "event": "test", "baseline_from_training": 0.5,
                "rows": [{"id": "1", "prediction": 1.0, "observed": 1},
                         {"id": "2", "prediction": 0.0, "observed": 0}]}
        result = evaluate(data)
        self.assertEqual(result["brier"], 0)
        self.assertEqual(result["baseline_brier"], 0.25)
        data["rows"][0]["observed"] = True
        with self.assertRaises(ValueError):
            evaluate(data)

    def test_mocked_http(self):
        opener = MagicMock()
        opener.open.return_value.__enter__.return_value.read.return_value = json.dumps(self.body).encode()
        with patch.dict("os.environ", {"TYPESAFE_API_KEY": "test-only"}), patch(
                "urllib.request.build_opener", return_value=opener):
            validate_answers(jev(self.state, self.qs, "jev-latest"), self.qs)
        request = opener.open.call_args.args[0]
        self.assertEqual(request.full_url, "https://api.typesafe.ai/v1/systemone")
        self.assertEqual(json.loads(request.data)["questions"], self.qs)
        self.assertEqual(opener.open.call_count, 1)

    def test_http_error_sanitized(self):
        opener = MagicMock()
        opener.open.side_effect = urllib.error.HTTPError("url", 429, "SECRET", {}, io.BytesIO(b"SECRET"))
        with patch.dict("os.environ", {"TYPESAFE_API_KEY": "test-only"}), patch(
                "urllib.request.build_opener", return_value=opener):
            with self.assertRaisesRegex(RuntimeError, "^http_429$"):
                jev(self.state, self.qs, "jev-latest")

    def test_redirect_refused(self):
        self.assertIsNone(NoRedirect().redirect_request(None))


if __name__ == "__main__":
    unittest.main()
