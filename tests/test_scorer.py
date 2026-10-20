"""Tests for score parsing, title exclusion and preference caps."""

import pytest

from applypilot.scoring import scorer

PREFS = {
    "base_location": "New York, NY, USA",
    "preferred_locations": ["New York, NY"],
    "seniority": ["mid", "senior"],
    "work_modes": ["remote_global", "remote_country", "hybrid", "onsite"],
}

RESPONSE = """SCORE: 8
WORK_MODE: onsite
LOCATION_MATCH: no
SENIORITY: senior
KEYWORDS: Python, AWS
REASONING: Strong backend match."""


def test_parse_reads_all_fields() -> None:
    r = scorer._parse_score_response(RESPONSE)
    assert r["score"] == 8
    assert (r["work_mode"], r["location_match"], r["seniority"]) == ("onsite", "no", "senior")
    assert r["keywords"] == "Python, AWS"


def test_unknown_labels_become_none() -> None:
    r = scorer._parse_score_response("SCORE: 6\nWORK_MODE: unknown\nSENIORITY: [mid]")
    assert r["work_mode"] is None
    assert r["seniority"] == "mid"


def test_onsite_elsewhere_is_capped() -> None:
    r = scorer.apply_preferences(scorer._parse_score_response(RESPONSE), PREFS)
    assert r["score"] == 3
    assert "location not workable" in r["reasoning"]


def test_wrong_level_is_capped() -> None:
    r = scorer.apply_preferences(
        {"score": 9, "reasoning": "x", "work_mode": "remote_global",
         "location_match": "yes", "seniority": "junior"}, PREFS)
    assert r["score"] == 4


def test_unwanted_work_mode_is_capped() -> None:
    prefs = {**PREFS, "work_modes": ["remote_global", "remote_country"]}
    r = scorer.apply_preferences(
        {"score": 9, "reasoning": "x", "work_mode": "hybrid",
         "location_match": "yes", "seniority": "senior"}, prefs)
    assert r["score"] == 3


def test_good_match_and_unknowns_are_not_capped() -> None:
    for labels in ({"work_mode": "remote_global", "location_match": "yes", "seniority": "senior"},
                   {"work_mode": None, "location_match": None, "seniority": None}):
        r = scorer.apply_preferences({"score": 9, "reasoning": "x", **labels}, PREFS)
        assert r["score"] == 9


def test_no_prefs_means_no_caps() -> None:
    # Without a base location there is nothing to judge location against.
    r = scorer.apply_preferences(scorer._parse_score_response(RESPONSE), {})
    assert r["score"] == 8


def test_markdown_decorated_reply_is_parsed() -> None:
    r = scorer._parse_score_response("**SCORE:** 7\n- **SENIORITY:** senior")
    assert (r["score"], r["seniority"]) == (7, "senior")


@pytest.mark.parametrize("title,expected", [
    ("Software Engineer Intern", "intern "),
    ("VP of Engineering", "vp "),
    ("Senior Software Engineer", None),
    ("Internal Tools Engineer", None),
])
def test_title_exclusion(title: str, expected: str | None) -> None:
    assert scorer.title_excluded(title, ["intern ", "vp ", "principal"]) == expected


def test_prompt_includes_candidate_context() -> None:
    prompt = scorer.build_score_prompt(PREFS)
    assert "New York, NY, USA" in prompt
    assert "{context}" not in scorer.build_score_prompt({})


def test_unparseable_reply_is_saved_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    class Fake:
        def chat(self, *a, **k):
            return "I cannot score this."

    monkeypatch.setattr(scorer, "get_client", lambda task: Fake())
    job = {"title": "SWE", "site": "Acme", "full_description": "..."}
    r = scorer.score_job("resume", job, PREFS)
    assert r["score"] == 0 and not r.get("error")  # saved, so it leaves the queue


def test_llm_failure_is_retried_next_run(monkeypatch: pytest.MonkeyPatch) -> None:
    class Fake:
        def chat(self, *a, **k):
            raise RuntimeError("overloaded")

    monkeypatch.setattr(scorer, "get_client", lambda task: Fake())
    job = {"title": "SWE", "site": "Acme", "full_description": "..."}
    assert scorer.score_job("resume", job, PREFS).get("error") is True


@pytest.mark.parametrize(("line", "expected"), [
    ("SALARY_MAX: 185000", 185000),
    ("SALARY_MAX: $142,500", 142500),
    ("**SALARY_MAX:** 150k", 150000),
    ("SALARY_MAX: unknown", None),
    ("SALARY_MAX: 185", None),  # ambiguous, ignored
    ("SALARY_MAX: 145000\nKEYWORDS: Java", 145000),  # next line's K is not "thousand"
])
def test_salary_parsing(line: str, expected: int | None) -> None:
    assert scorer._parse_score_response(f"SCORE: 8\n{line}")["salary_max"] == expected


def test_low_posted_salary_is_capped_but_unknown_is_not() -> None:
    prefs = {**PREFS, "min_salary": 160000}
    base = {"score": 9, "reasoning": "x", "work_mode": "remote_global",
            "location_match": "yes", "seniority": "mid"}
    assert scorer.apply_preferences({**base, "salary_max": 140000}, prefs)["score"] == 5
    assert scorer.apply_preferences({**base, "salary_max": 175000}, prefs)["score"] == 9
    assert scorer.apply_preferences({**base, "salary_max": None}, prefs)["score"] == 9
