"""Tests for honest titles and the cover-letter fabrication judge."""

import pytest

from applypilot.scoring import cover_letter
from applypilot.scoring.tailor import honest_title

MID_LEVEL = "Software Engineer, Visa Inc.\nSoftware Engineer, Goldman Sachs"


@pytest.mark.parametrize(("title", "expected"), [
    ("Senior Backend Engineer (Java/Spring Boot)", "Backend Engineer (Java/Spring Boot)"),
    ("Sr. Software Engineer", "Software Engineer"),
    ("Senior/Staff Software Engineer", "Software Engineer"),
    ("Backend Engineer", "Backend Engineer"),
    ("Senior", "Software Engineer"),
])
def test_unearned_seniority_is_dropped(title: str, expected: str) -> None:
    assert honest_title(title, MID_LEVEL) == expected


def test_earned_seniority_is_kept() -> None:
    assert honest_title("Senior Backend Engineer", "Senior Software Engineer, Acme") == \
        "Senior Backend Engineer"


LETTER = "Dear Hiring Manager,\n\nI built a Kafka testing framework at Visa.\n\nJames"


class FakeClient:
    def __init__(self, replies: dict[str, list[str]]):
        self.replies = replies

    def for_task(self, task: str):
        replies = self.replies[task]
        client = self

        class C:
            def chat(self, *a, **k):
                return replies.pop(0) if len(replies) > 1 else replies[0]
        return C()


def _patch(monkeypatch: pytest.MonkeyPatch, judge: list[str]) -> None:
    fake = FakeClient({"cover": [LETTER], "judge": judge})
    monkeypatch.setattr(cover_letter, "get_client", fake.for_task)
    monkeypatch.setattr(cover_letter, "validate_cover_letter",
                        lambda letter, mode: {"passed": True, "errors": []})


def test_letter_passing_judge_is_returned(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, ["VERDICT: PASS\nISSUES: none"])
    assert cover_letter.generate_cover_letter("resume", {"title": "SWE", "site": "x"}, {}) == LETTER


def test_letter_judge_retries_then_passes(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, ["VERDICT: FAIL\nISSUES: invents Jenkins", "VERDICT: PASS\nISSUES: none"])
    assert cover_letter.generate_cover_letter("resume", {"title": "SWE", "site": "x"}, {}) == LETTER


def test_letter_always_rejected_is_not_saved(monkeypatch: pytest.MonkeyPatch) -> None:
    _patch(monkeypatch, ["VERDICT: FAIL\nISSUES: invents Jenkins"])
    with pytest.raises(ValueError, match="Jenkins"):
        cover_letter.generate_cover_letter("resume", {"title": "SWE", "site": "x"}, {})
