"""Deterministic stand-in for the Jev API. Test-only; not a behavioral model."""
import hashlib
import json

MODEL = "fake-jev-for-tests"


def study_fixture(state, qs):
    # Invented rules: only the budget and whether the copy says "leakproof".
    specific = "leakproof" in state["stimulus"]["text"].lower()
    over_budget = state["profile"]["context"]["budget_usd"] < 39
    engage = 0.18 if over_budget else (0.50 if specific else 0.28)
    action = {"ignore": 0.15, "engage": engage, "research": 0.60 - engage,
              "defer": 0.15, "unknown": 0.10}
    sentiment = ({"positive": 0.25, "mixed": 0.55, "negative": 0.10, "unknown": 0.10}
                 if over_budget else {"positive": 0.60, "mixed": 0.25, "negative": 0.05, "unknown": 0.10})
    answers = {k: {"type": "choice", "choice": max(ps, key=ps.get), "probabilities": ps}
               for k, ps in (("action", action), ("sentiment", sentiment))}
    for k, p in {"relevant": 0.90 if specific else 0.45, "price_objection": 0.99 if over_budget else 0.01,
                 "proof_gap": 0.80, "confusion": 0.10}.items():
        answers[k] = {"type": "noul", "noul": p}
    return {"model": MODEL, "answers": answers}


# Handwritten labels for the exact invented example transcripts, keyed by state digest.
DEMO_LABELS = {'59c47f39e745ca8998c6a2f494074dd5ddc0f8c3eb9265a509ff0d72693ce493': {'sentiment': 'mixed', 'likely_behavior': 'defer', 'behavior_basis': 'stated_intent', 'main_barrier': 'price', 'research_next_step': 'test_price_value', 'evidence_sentiment': 't2', 'evidence_likely_behavior': 't5', 'evidence_main_barrier': 't2', 'evidence_research_next_step': 't2'}, 'a352eab7108e9240cce0e1bdedc8a6a3d1e7bba0bb5ee8a8fdd9a8981cbb8efb': {'sentiment': 'mixed', 'likely_behavior': 'investigate', 'behavior_basis': 'stated_intent', 'main_barrier': 'proof', 'research_next_step': 'test_proof', 'evidence_sentiment': 't3', 'evidence_likely_behavior': 't6', 'evidence_main_barrier': 't3', 'evidence_research_next_step': 't3'}, '1d24f84e58e458c1a838ca6eda8205355f932de1dc2f0d50a074b359ad7afa94': {'sentiment': 'negative', 'likely_behavior': 'reject', 'behavior_basis': 'both', 'main_barrier': 'fit', 'research_next_step': 'investigate_fit', 'evidence_sentiment': 't4', 'evidence_likely_behavior': 't4', 'evidence_main_barrier': 't2', 'evidence_research_next_step': 't2'}}


def interview_fixture(state, questions):
    digest = hashlib.sha256(json.dumps(state, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    labels = DEMO_LABELS[digest]
    answers = {}
    for key, question in questions.items():
        if question['type'] == 'noul':
            answers[key] = {"type": "noul", "noul": 0.9}
        else:
            selected = labels[key]
            options = question['criteria']
            probabilities = {option: 0.8 if option == selected else 0.2 / (len(options) - 1) for option in options}
            answers[key] = {"type": "choice", "choice": selected, "probabilities": probabilities}
    return {"model": MODEL, "answers": answers}


def respond(state, questions):
    return interview_fixture(state, questions) if "turns" in state else study_fixture(state, questions)


class _Response:
    def __init__(self, data):
        self._data = data

    def read(self, limit=-1):
        return self._data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _Opener:
    def open(self, request, timeout=None):
        payload = json.loads(request.data)
        return _Response(json.dumps(respond(payload["state"], payload["questions"])).encode())


def fake_build_opener(*handlers):
    """Drop-in for urllib.request.build_opener that answers like the old demo."""
    return _Opener()
