"""Tests for LLM provider selection and the Claude Code CLI client."""

import json
import subprocess

import pytest

from applypilot import config, llm

PROVIDER_VARS = ("LLM_PROVIDER", "LLM_URL", "GEMINI_API_KEY", "OPENAI_API_KEY", "LLM_MODEL")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch, tmp_path):
    for var in PROVIDER_VARS:
        monkeypatch.delenv(var, raising=False)
    for task in llm.CLAUDE_TASK_MODELS:
        monkeypatch.delenv(f"CLAUDE_MODEL_{task.upper()}", raising=False)
    monkeypatch.setattr(config, "get_claude_path", lambda: "C:/bin/claude.exe")
    monkeypatch.setattr(config, "LOG_DIR", tmp_path)
    monkeypatch.setattr(llm, "_instance", None)
    monkeypatch.setattr(llm, "_claude_clients", {})


def _cli_json(**fields) -> str:
    data = {"type": "result", "subtype": "success", "is_error": False, "result": "ok"}
    data.update(fields)
    return json.dumps(data)


# -- provider selection -------------------------------------------------------

def test_claude_is_the_fallback_when_no_api_key() -> None:
    assert config.llm_provider() == "claude"


def test_api_key_wins_over_claude(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert config.llm_provider() == "gemini"


def test_llm_provider_forces_claude(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("LLM_PROVIDER", "Claude")
    assert config.llm_provider() == "claude"


def test_no_provider_without_claude(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(config, "get_claude_path", lambda: None)
    assert config.llm_provider() is None


@pytest.mark.parametrize("forced", ["claude", "gemini", "openai", "local"])
def test_forced_provider_must_be_usable(monkeypatch: pytest.MonkeyPatch, forced: str) -> None:
    # Otherwise the tier gate passes and every LLM call fails.
    monkeypatch.setattr(config, "get_claude_path", lambda: None)
    monkeypatch.setenv("LLM_PROVIDER", forced)
    assert config.llm_provider() is None


# -- per-task models ----------------------------------------------------------

def test_cheap_model_for_bulk_tasks_and_sonnet_for_writing() -> None:
    assert llm.get_client("score").model == "haiku"
    assert llm.get_client("extract").model == "haiku"
    assert llm.get_client("tailor").model == "sonnet"
    assert llm.get_client("judge").model == "sonnet"
    assert llm.get_client("cover").model == "sonnet"


def test_usage_is_logged_per_task(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(
        cmd, 0, stdout=_cli_json(total_cost_usd=0.01), stderr=""))
    llm.get_client("tailor").ask("a")
    llm.get_client("judge").ask("b")  # same model, separate row
    assert set(llm.usage_summary(1)) == {"tailor", "judge"}


def test_task_override_beats_global_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CLAUDE_MODEL", "opus")
    monkeypatch.setenv("CLAUDE_MODEL_SCORE", "haiku")
    assert llm.claude_model_for("score") == "haiku"
    assert llm.claude_model_for("tailor") == "opus"


def test_api_model_name_does_not_leak_into_claude(monkeypatch: pytest.MonkeyPatch) -> None:
    # An older .env from the Gemini setup still has LLM_MODEL.
    monkeypatch.setenv("LLM_MODEL", "gemini-2.0-flash")
    assert llm.claude_model_for("score") == "haiku"


# -- command line -------------------------------------------------------------

def test_cmd_is_locked_down() -> None:
    cmd = llm.build_claude_cmd("claude", "haiku", "sys.md")
    assert cmd[cmd.index("--tools") + 1] == ""
    for flag in ("--safe-mode", "--strict-mcp-config", "--disable-slash-commands",
                 "--no-session-persistence"):
        assert flag in cmd
    assert cmd[cmd.index("--system-prompt-file") + 1] == "sys.md"
    assert "bypassPermissions" not in cmd
    assert "--dangerously-skip-permissions" not in cmd


def test_system_and_user_messages_are_split() -> None:
    system, prompt = llm._render_messages([
        {"role": "system", "content": "Be strict."},
        {"role": "user", "content": "Score this."},
    ])
    assert (system, prompt) == ("Be strict.", "Score this.")


def test_none_content_is_tolerated() -> None:
    _, prompt = llm._render_messages([
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": None},
        {"role": "user", "content": "again"},
    ])
    assert "[ASSISTANT]" in prompt and "again" in prompt


# -- output parsing -----------------------------------------------------------

def test_parse_success() -> None:
    assert llm.parse_claude_output("noise\n" + _cli_json(result="SCORE: 8")) == "SCORE: 8"


@pytest.mark.parametrize("message", [
    "Claude AI usage limit reached|1760000000",
    "You've hit your limit · resets 3pm (America/Chicago)",
])
def test_usage_limit_is_its_own_error(message: str) -> None:
    out = _cli_json(subtype="error", is_error=True, api_error_status=429, result=message)
    with pytest.raises(llm.ClaudeUsageLimitError):
        llm.parse_claude_output(out)


def test_other_errors_raise_runtime_error() -> None:
    with pytest.raises(RuntimeError) as exc:
        llm.parse_claude_output(_cli_json(subtype="error", is_error=True, result="overloaded"))
    assert not isinstance(exc.value, llm.ClaudeUsageLimitError)


def test_every_call_is_logged_with_its_cost() -> None:
    llm.parse_claude_output(_cli_json(total_cost_usd=0.01, usage={"input_tokens": 100}), kind="haiku")
    with pytest.raises(RuntimeError):
        llm.parse_claude_output(_cli_json(subtype="error", is_error=True, total_cost_usd=0.02,
                                          result="overloaded"), kind="haiku")
    llm.record_usage("apply", {"total_cost_usd": 1.5, "usage": {"output_tokens": 50}})

    totals = llm.usage_summary(days=1)
    assert totals["haiku"]["calls"] == 2
    assert totals["haiku"]["cost_usd"] == pytest.approx(0.03)
    assert totals["apply"] == {"calls": 1, "cost_usd": 1.5, "tokens": 50}


def test_garbage_output_raises() -> None:
    with pytest.raises(RuntimeError):
        llm.parse_claude_output("not json")


# -- client -------------------------------------------------------------------

def test_chat_passes_prompt_on_stdin_and_cleans_up(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict = {}

    def fake_run(cmd, **kwargs):
        sp_file = cmd[cmd.index("--system-prompt-file") + 1]
        with open(sp_file, encoding="utf-8") as fh:
            seen["system"] = fh.read()
        seen["sp_file"] = sp_file
        seen["input"] = kwargs["input"]
        seen["env"] = kwargs["env"]
        return subprocess.CompletedProcess(cmd, 0, stdout=_cli_json(result="SCORE: 9"), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setenv("CLAUDECODE", "1")
    client = llm.ClaudeCLIClient("claude", "haiku")
    reply = client.chat([{"role": "system", "content": "rubric"},
                         {"role": "user", "content": "job text"}])

    assert reply == "SCORE: 9"
    assert seen["system"] == "rubric"
    assert seen["input"] == "job text"
    assert "CLAUDECODE" not in seen["env"]
    assert not __import__("os").path.exists(seen["sp_file"])


def test_usage_limit_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return subprocess.CompletedProcess(
            cmd, 1, stdout=_cli_json(subtype="error", is_error=True,
                                     result="Claude AI usage limit reached"), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(llm.ClaudeUsageLimitError):
        llm.ClaudeCLIClient("claude", "haiku").ask("hi")
    assert len(calls) == 1


def test_timeout_is_retried_only_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    with pytest.raises(subprocess.TimeoutExpired):
        llm.ClaudeCLIClient("claude", "haiku").ask("hi")
    assert len(calls) == 2


def test_transient_error_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    outputs = iter([_cli_json(subtype="error", is_error=True, result="overloaded"),
                    _cli_json(result="fine")])
    monkeypatch.setattr(subprocess, "run", lambda cmd, **kw: subprocess.CompletedProcess(
        cmd, 0, stdout=next(outputs), stderr=""))
    monkeypatch.setattr(llm.time, "sleep", lambda s: None)
    assert llm.ClaudeCLIClient("claude", "haiku").ask("hi") == "fine"
