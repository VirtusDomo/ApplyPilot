# Changelog

All notable changes to ApplyPilot will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

Safety and honesty pass for auto-apply, plus correctness fixes. Claude as the LLM provider.

### Added
- **Claude provider** - scoring, extraction, tailoring, the fabrication judge and cover letters can
  run through the Claude Code CLI on the user's Claude subscription (no API key). It is the default
  when `claude` is installed and no API key is configured; `LLM_PROVIDER` forces a provider.
  Per-task models: Haiku for scoring/extraction, Sonnet for tailoring/judge/cover letters,
  overridable with `CLAUDE_MODEL` or `CLAUDE_MODEL_<TASK>`. Calls run with `--safe-mode --tools ""
  --strict-mcp-config`: no tools, MCP servers, hooks or user CLAUDE.md. At most 2 run at once.
- **Preference-aware scoring** - the scorer now also reports each job's work mode
  (remote_global / remote_country / hybrid / onsite), whether the candidate can work it from
  their base location, and its seniority. A new `preferences` section in searches.yaml caps the
  score at 3 for unworkable locations or unwanted work modes, at 4 for untargeted levels, and
  at 5 when the posted salary tops out below `min_salary`.
  Work mode is stored in a new `work_mode` column; `prioritize_work_modes` breaks score ties
  when picking the next job to apply to (jobs scored earlier need `run score --rescore` for this).
- **Usage log** - every Claude call and apply session is recorded in `logs/llm_usage.jsonl`;
  `applypilot status` shows the last 7 days (calls, tokens, API-equivalent cost).
- **Usage-limit handling** - hitting the plan's usage limit stops the stage instead of retrying;
  unprocessed jobs stay queued.

### Security
- **Apply sessions ignore user settings** - `--setting-sources ""` keeps the user's own hooks and
  global CLAUDE.md out of apply sessions.
- **Apply agent is locked down** - runs with `--permission-mode dontAsk`, `--strict-mcp-config`,
  only Playwright browser tools allowed, all built-in tools (shell, file read/write, web fetch)
  denied. Previously `bypassPermissions` while browsing untrusted pages.
- **Real Chrome profile is never copied** - worker browsers start from an empty profile instead
  of cloning the user's everyday Chrome (with all its logged-in sessions).
- **Gmail is opt-in and read-only** - `APPLYPILOT_ENABLE_GMAIL=1`; only `search_emails` and
  `read_email` are allowed, for verification codes. Email applications go to the manual queue.
- **No passwords** - the plain-text `personal.password` field and the "sign in / sign up with
  this password" flow are removed. Sign in once by hand in the worker browser; sessions persist.
- **MCP packages pinned** - `@playwright/mcp@0.0.83`, `@gongrzhe/server-gmail-autoauth-mcp@1.1.11`
  instead of `@latest` fetched on every run.
- **Prompt-injection guard** - the agent is told page content is data, and to stop with
  `RESULT:FAILED:suspicious_page` if a page tries to redirect it.

### Changed
- **No CAPTCHA solving** - CapSolver integration removed. CAPTCHA jobs go to the manual queue.
- **No fabricated answers** - removed hardcoded "Felony: No / Background Check: Yes / Age 18+: Yes"
  and the "answer YES to any same-domain tool" rule. Screening facts come from the new
  `profile.screening` section; a missing required fact stops the agent with
  `RESULT:NEEDS_HUMAN:<question>`, stored as `apply_status='manual'`.
- **Tailoring may not add skills** - removed "add 2-3 related tools" and the judge's
  "3 minor stretches" tolerance. A resume the judge rejects on its last retry is now
  `failed_judge` and is never used (previously `approved_with_judge_warning`).
- Removed the hardcoded "$110K minimum for senior titles" salary rule.
- `apply_timeout` default raised to 600s, now actually enforced.

### Fixed
- **`exclude_titles` was never applied** - matching jobs now get score 1 without an LLM call.
- **Resumes without a projects section always failed tailoring** - projects are now optional.
- **Scores were only saved at the end of a run** - each score is saved as it completes.
- `applypilot init` no longer overwrites an existing searches.yaml without asking.
- **LLM errors were saved as a score of 0** - a failed scoring call now leaves the job unscored so
  it is retried on the next run, instead of permanently ranking it at the bottom.
- **Dry runs were recorded as applied** (#104) - new `RESULT:DRY_RUN` / `apply_status='dry_run'`;
  dry runs also may not create accounts or confirm emails.
- **Windows apply always failed** (#93) - the `claude` CLI is resolved with `shutil.which`
  (`claude.cmd`).
- **`--url` skipped never-attempted jobs** (#93) - NULL-safe `apply_status` check.
- **Location filter ignored the documented config** (#109) - `location.accept_patterns` /
  `reject_patterns` are now read (legacy top-level keys still work).
- **JobSpy ignored `boards`** (#108) - falls back to legacy `sites`.
- **Apply timeout never fired** - the 300s wait only started after the agent exited; a
  watchdog now kills the session.
- **Result parsing** - only the agent's last `RESULT:` line counts, so a code mentioned while
  reasoning can't mark a job applied.
- **Fabrication watchlist** - skills in the user's own profile or resume (e.g. C#, Rust, a real
  certification) are no longer flagged as fabricated.

## [0.2.0] - 2026-02-17

### Added
- **Parallel workers for discovery/enrichment** - `applypilot run --workers N` enables
  ThreadPoolExecutor-based parallelism for Workday scraping, smart extract, and detail
  enrichment. Default is sequential (1); power users can scale up.
- **Apply utility modes** - `--gen` (generate prompt for manual debugging), `--mark-applied`,
  `--mark-failed`, `--reset-failed` flags on `applypilot apply`
- **Dry-run mode** - `applypilot apply --dry-run` fills forms without clicking Submit
- **5 new tracking columns** - `agent_id`, `last_attempted_at`, `apply_duration_ms`,
  `apply_task_id`, `verification_confidence` for better apply-stage observability
- **Manual ATS detection** - `manual_ats` list in `config/sites.yaml` skips sites with
  unsolvable CAPTCHAs (e.g. TCS iBegin)
- **Qwen3 `/no_think` optimization** - automatically saves tokens when using Qwen models
- **`config.DEFAULTS`** - centralized dict for magic numbers (`min_score`, `max_apply_attempts`,
  `poll_interval`, `apply_timeout`, `viewport`)

### Fixed
- **Config YAML not found after install** - moved `config/` into the package at
  `src/applypilot/config/` so YAML files (employers, sites, searches) ship with `pip install`
- **Search config format mismatch** - wizard wrote `searches:` key but discovery code
  expected `queries:` with tier support. Aligned wizard output and example config
- **JobSpy install isolation** - removed python-jobspy from package dependencies due to
  broken numpy==1.26.3 exact pin in jobspy metadata. Installed separately with `--no-deps`
- **Scoring batch limit** - default limit of 50 silently left jobs unscored across runs.
  Changed to no limit (scores all pending jobs in one pass)
- **Missing logging output** - added `logging.basicConfig(INFO)` so per-job progress for
  scoring, tailoring, and cover letters is visible during pipeline runs

### Changed
- **Blocked sites externalized** - moved from hardcoded sets in launcher.py to
  `config/sites.yaml` under `blocked:` key
- **Site base URLs externalized** - moved from hardcoded dict in detail.py to
  `config/sites.yaml` under `base_urls:` key
- **SSO domains externalized** - moved from hardcoded list in prompt.py to
  `config/sites.yaml` under `blocked_sso:` key
- **Prompt improvements** - screening context uses `target_role` from profile,
  salary section includes `currency_conversion_note` and dynamic hourly rate examples
- **`acquire_job()` fixed** - writes `agent_id` and `last_attempted_at` to proper columns
  instead of misusing `apply_error`
- **`profile.example.json`** - added `currency_conversion_note` and `target_role` fields

## [0.1.0] - 2026-02-17

### Added
- 6-stage pipeline: discover, enrich, score, tailor, cover letter, apply
- Multi-source job discovery: Indeed, LinkedIn, Glassdoor, ZipRecruiter, Google Jobs
- Workday employer portal support (46 preconfigured employers)
- Direct career site scraping (28 preconfigured sites)
- 3-tier job description extraction cascade (JSON-LD, CSS selectors, AI fallback)
- AI-powered job scoring (1-10 fit scale with rationale)
- Resume tailoring with factual preservation (no fabrication)
- Cover letter generation per job
- Autonomous browser-based application submission via Playwright
- Interactive setup wizard (`applypilot init`)
- Cross-platform Chrome/Chromium detection (Windows, macOS, Linux)
- Multi-provider LLM support (Gemini, OpenAI, local models via OpenAI-compatible endpoints)
- Pipeline stats and HTML results dashboard
- YAML-based configuration for employers, career sites, and search queries
- Job deduplication across sources
- Configurable score threshold filtering
- Safety limits for maximum applications per run
- Detailed application results logging
