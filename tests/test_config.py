"""Tests for search-config parsing."""

from applypilot.config import load_location_filter


def test_nested_location_config_is_read() -> None:
    # The documented format in searches.example.yaml (issue #109).
    cfg = {"location": {"accept_patterns": ["Austin"], "reject_patterns": ["onsite only"]}}
    assert load_location_filter(cfg) == (["Austin"], ["onsite only"])


def test_legacy_top_level_keys_still_work() -> None:
    cfg = {"location_accept": ["Toronto"], "location_reject_non_remote": ["India"]}
    assert load_location_filter(cfg) == (["Toronto"], ["India"])


def test_missing_config_gives_empty_lists() -> None:
    assert load_location_filter({}) == ([], [])


def test_job_company_prefers_employer_over_board() -> None:
    from applypilot.database import job_company
    assert job_company({"site": "linkedin", "company": "Gen Digital"}) == "Gen Digital"
    assert job_company({"site": "Acme Careers", "company": None}) == "Acme Careers"
