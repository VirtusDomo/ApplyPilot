"""Tests for apply-stage result parsing, manual routing and the locked-down agent command."""

from pathlib import Path

import pytest

from applypilot.apply import launcher


@pytest.mark.parametrize(
    ("output", "dry_run", "expected"),
    [
        ("Filled everything.\nRESULT:APPLIED", False, "applied"),
        ("RESULT:EXPIRED", False, "expired"),
        ("RESULT:CAPTCHA", False, "captcha"),
        ("RESULT:FAILED:captcha", False, "captcha"),
        ("RESULT:FAILED:not_eligible_location", False, "failed:not_eligible_location"),
        ("RESULT:FAILED:stuck**", False, "failed:stuck"),
        ("RESULT:NEEDS_HUMAN:Do you hold a CPA license?", False,
         "needs_human:Do you hold a CPA license?"),
        ("no code at all", False, "failed:no_result_line"),
        ("RESULT:DRY_RUN\n- salary field skipped", True, "dry_run"),
    ],
)
def test_parse_result(output: str, dry_run: bool, expected: str) -> None:
    assert launcher.parse_result(output, dry_run=dry_run) == expected


def test_parse_result_uses_last_code_only() -> None:
    # A code mentioned while reasoning must not decide the outcome.
    output = "I will output RESULT:APPLIED once I see the confirmation.\nRESULT:FAILED:page_error"
    assert launcher.parse_result(output) == "failed:page_error"


def test_dry_run_can_never_be_applied() -> None:
    assert launcher.parse_result("RESULT:APPLIED", dry_run=True) == "dry_run"


@pytest.mark.parametrize(
    ("result", "manual"),
    [
        ("needs_human:email_application to jobs@x.com", True),
        ("captcha", True),
        ("failed:account_required", True),
        ("failed:sso_required", True),
        ("failed:stuck", False),
        ("applied", False),
    ],
)
def test_needs_manual(result: str, manual: bool) -> None:
    assert launcher._needs_manual(result) is manual


def test_claude_cmd_is_locked_down(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("APPLYPILOT_ENABLE_GMAIL", raising=False)
    cmd = launcher._build_claude_cmd("claude.cmd", "sonnet", Path("mcp.json"))
    assert "bypassPermissions" not in cmd
    assert cmd[cmd.index("--permission-mode") + 1] == "dontAsk"
    assert "--strict-mcp-config" in cmd
    assert cmd[cmd.index("--setting-sources") + 1] == ""  # no user hooks or CLAUDE.md
    assert cmd[cmd.index("--allowedTools") + 1] == "mcp__playwright"
    denied = cmd[cmd.index("--disallowedTools") + 1].split(",")
    for tool in ("Bash", "PowerShell", "Read", "Write", "WebFetch"):
        assert tool in denied


def test_gmail_is_opt_in_and_read_only(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("APPLYPILOT_ENABLE_GMAIL", raising=False)
    assert "gmail" not in launcher._make_mcp_config(9222)["mcpServers"]

    monkeypatch.setenv("APPLYPILOT_ENABLE_GMAIL", "1")
    assert "gmail" in launcher._make_mcp_config(9222)["mcpServers"]
    allowed = launcher._build_claude_cmd("claude", "sonnet", Path("m.json"))
    allowed_tools = allowed[allowed.index("--allowedTools") + 1].split(",")
    assert set(allowed_tools) == {"mcp__playwright", "mcp__gmail__search_emails", "mcp__gmail__read_email"}


def test_mcp_packages_are_pinned(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APPLYPILOT_ENABLE_GMAIL", "1")
    for server in launcher._make_mcp_config(9222)["mcpServers"].values():
        package = next(a for a in server["args"] if a.startswith("@"))
        assert "@latest" not in package
        assert package.rsplit("@", 1)[1][0].isdigit(), package


@pytest.mark.parametrize(("priority", "expected"), [
    ([], "https://jobs.example.com/a"),
    (["remote_global"], "https://jobs.example.com/z"),
    ("remote_global", "https://jobs.example.com/z"),  # a bare string works too
])
def test_equal_scores_apply_to_preferred_work_mode_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, priority: list, expected: str,
) -> None:
    from applypilot import config, database

    conn = database.init_db(tmp_path / "jobs.db")
    for url, mode in (("https://jobs.example.com/a", "onsite"),
                      ("https://jobs.example.com/z", "remote_global")):
        conn.execute(
            "INSERT INTO jobs (url, title, site, fit_score, work_mode, tailored_resume_path) "
            "VALUES (?, 'SWE', 'Example', 8, ?, 'r.txt')", (url, mode))
    conn.commit()
    monkeypatch.setattr(launcher, "get_connection", lambda: conn)
    monkeypatch.setattr(launcher, "_load_blocked", lambda: ([], []))
    monkeypatch.setattr(config, "load_search_config",
                        lambda: {"preferences": {"prioritize_work_modes": priority}})

    assert launcher.acquire_job(min_score=7)["url"] == expected
