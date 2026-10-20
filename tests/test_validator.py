"""Tests for fabrication detection in tailored resumes."""

from applypilot.scoring.validator import _fabricated_terms, validate_json_fields

PROFILE = {
    "skills_boundary": {"languages": ["C#", "Python"], "tools": ["Docker"]},
    "resume_facts": {"preserved_companies": ["Acme"], "preserved_school": "State U"},
}


def _resume(skills: str) -> dict:
    return {
        "title": "Backend Engineer",
        "summary": "Builds services.",
        "skills": {"Languages": skills},
        "experience": [{"header": "Engineer at Acme", "bullets": ["Built APIs"]}],
        "projects": [{"header": "Tool", "bullets": ["Wrote a CLI"]}],
        "education": "State U | BSc",
    }


def test_own_skills_are_not_fabricated() -> None:
    # Previously every C# developer was flagged for listing C#.
    result = validate_json_fields(_resume("C#, Python"), PROFILE)
    assert result["passed"], result["errors"]


def test_invented_skill_is_flagged() -> None:
    result = validate_json_fields(_resume("Python, Rust"), PROFILE)
    assert not result["passed"]
    assert any("rust" in e for e in result["errors"])


def test_skill_in_original_resume_is_allowed() -> None:
    assert _fabricated_terms("python, golang", PROFILE, original_text="Shipped a Golang service") == []


def test_real_certification_is_allowed_only_when_claimed() -> None:
    assert "certified" in _fabricated_terms("AWS Certified", PROFILE)
    assert _fabricated_terms("AWS Certified", PROFILE, original_text="AWS Certified Developer") == []


def test_resume_without_projects_passes() -> None:
    # Many senior resumes have no projects section; tailoring used to fail every time.
    for projects in (None, []):
        data = _resume("Python")
        data["projects"] = projects
        result = validate_json_fields(data, PROFILE)
        assert result["passed"], result["errors"]
    data = _resume("Python")
    del data["projects"]
    assert validate_json_fields(data, PROFILE)["passed"]
