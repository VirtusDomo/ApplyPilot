"""Tests that the apply prompt never fabricates answers or carries secrets."""

from pathlib import Path

import pytest

from applypilot import config
from applypilot.apply import prompt

PROFILE = {
    "personal": {"full_name": "Sam Lee", "email": "sam@example.com", "phone": "555-0100",
                 "city": "Austin", "country": "USA", "password": "hunter2"},
    "work_authorization": {"legally_authorized_to_work": "Yes", "require_sponsorship": "No"},
    "compensation": {"salary_expectation": "90000"},
    "skills_boundary": {"languages": ["Python"]},
    "screening": {"age_18_or_older": "Yes"},
}


@pytest.fixture
def build(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    resume = tmp_path / "resume.txt"
    resume.with_suffix(".pdf").write_bytes(b"%PDF-1.4")
    monkeypatch.setattr(config, "load_profile", lambda: PROFILE)
    monkeypatch.setattr(config, "load_search_config", dict)
    monkeypatch.setattr(config, "APPLY_WORKER_DIR", tmp_path / "workers")
    monkeypatch.setenv("CAPSOLVER_API_KEY", "secret-capsolver-key")
    monkeypatch.delenv("APPLYPILOT_ENABLE_GMAIL", raising=False)
    job = {"url": "https://jobs.example.com/1", "title": "Engineer", "site": "Example",
           "tailored_resume_path": str(resume)}

    def _build(dry_run: bool = False) -> str:
        return prompt.build_prompt(job=job, tailored_resume="resume text", dry_run=dry_run)

    return _build


def test_no_hardcoded_attestations(build) -> None:
    text = build()
    assert "Felony: No" not in text
    assert "Background Check: Yes" not in text
    assert "Age 18+: Yes" in text  # stated in the profile
    assert "Criminal Convictions: NOT PROVIDED" in text
    assert "RESULT:NEEDS_HUMAN" in text


def test_no_secrets_or_captcha_solving(build) -> None:
    text = build()
    assert "hunter2" not in text
    assert "secret-capsolver-key" not in text
    assert "capsolver" not in text.lower()
    assert "Try sign up" not in text


def test_no_skill_inflation_rule(build) -> None:
    assert "answer YES" not in build()


def test_dry_run_never_reports_applied(build) -> None:
    text = build(dry_run=True)
    assert "RESULT:DRY_RUN" in text
    assert "then output RESULT:APPLIED" not in text
    assert "Do NOT create accounts" in text


def test_email_verification_needs_human_without_gmail(build) -> None:
    assert "RESULT:NEEDS_HUMAN:email_verification" in build()


def test_upload_copy_goes_to_agent_workdir(build, tmp_path: Path) -> None:
    # The browser tool only uploads files inside the agent's working directory.
    workdir = tmp_path / "worker-0"
    workdir.mkdir()
    text = prompt.build_prompt(job={"url": "u", "title": "Engineer", "site": "Example",
                                    "tailored_resume_path": str(tmp_path / "resume.txt")},
                               tailored_resume="resume text", upload_dir=workdir)
    assert (workdir / "Sam_Lee_Resume.pdf").exists()
    assert str(workdir / "Sam_Lee_Resume.pdf") in text


@pytest.mark.parametrize(("value", "expected"), [
    ("Two weeks' notice", "Two weeks' notice (if a date is required: 11/04/2026)"),
    ("3 weeks", "3 weeks (if a date is required: 11/11/2026)"),
    ("Immediately", "Immediately"),
])
def test_notice_period_gets_a_concrete_date(value: str, expected: str) -> None:
    from datetime import date
    assert prompt._start_date(value, today=date(2026, 10, 21)) == expected


def test_missing_onsite_answer_is_not_guessed(build) -> None:
    assert "Willing to Work Onsite/Hybrid: NOT PROVIDED" in build()
