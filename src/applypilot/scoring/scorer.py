"""Job fit scoring: LLM-powered evaluation of candidate-job match quality.

Scores jobs on a 1-10 scale by comparing the user's resume against each
job description. All personal data is loaded at runtime from the user's
profile and resume file.
"""

import json
import logging
import re
import time
from datetime import datetime, timezone

from applypilot.config import RESUME_PATH, load_profile, load_search_config
from applypilot.database import get_connection, get_jobs_by_stage
from applypilot.llm import ClaudeUsageLimitError, get_client

log = logging.getLogger(__name__)


# ── Scoring Prompt ────────────────────────────────────────────────────────

SCORE_PROMPT = """You are a job fit evaluator. Given a candidate's resume and a job description, score how well the candidate fits the role.

SCORING CRITERIA:
- 9-10: Perfect match. Candidate has direct experience in nearly all required skills and qualifications.
- 7-8: Strong match. Candidate has most required skills, minor gaps easily bridged.
- 5-6: Moderate match. Candidate has some relevant skills but missing key requirements.
- 3-4: Weak match. Significant skill gaps, would need substantial ramp-up.
- 1-2: Poor match. Completely different field or experience level.

IMPORTANT FACTORS:
- Weight technical skills heavily (programming languages, frameworks, tools)
- Consider transferable experience (automation, scripting, API work)
- Factor in the candidate's project experience
- Be realistic about experience level vs. job requirements (years of experience, seniority)
- SCORE is skills and experience fit only. Report location and seniority on their own lines below.
{context}
RESPOND IN EXACTLY THIS FORMAT (no other text):
SCORE: [1-10]
WORK_MODE: [remote_global | remote_country | hybrid | onsite | unknown]
LOCATION_MATCH: [yes | no | unknown]
SENIORITY: [junior | mid | senior | staff_plus | unknown]
SALARY_MAX: [top of the posted annual base salary range in USD as a plain number, or unknown]
KEYWORDS: [comma-separated ATS keywords from the job description that match or could match the candidate]
REASONING: [2-3 sentences explaining the score]

WORK_MODE: remote_global = fully remote and open to working from other countries; remote_country = fully remote but limited to specific countries or regions; hybrid = some days in an office; onsite = in an office full time. If the posting names an office location and never mentions remote or hybrid work, use onsite.
LOCATION_MATCH: yes if the candidate can do this job from where they are based (remote: their country is in the allowed region; hybrid/onsite: within commuting distance of one of their preferred locations).
SENIORITY: the level the posting is hiring for (junior = entry level or under 2 years; mid = about 2-5 years; senior = about 5+ years; staff_plus = staff, principal or above).
SALARY_MAX: only from a salary actually stated in the posting (convert hourly x 2080); never estimate."""

WORK_MODES = ("remote_global", "remote_country", "hybrid", "onsite")
SENIORITY_LEVELS = ("junior", "mid", "senior", "staff_plus")


def build_score_prompt(prefs: dict | None = None) -> str:
    """Fill the scoring prompt with the candidate context from searches.yaml ``preferences``."""
    prefs = prefs or {}
    lines = []
    if prefs.get("base_location"):
        lines.append(f"- Candidate is based in: {prefs['base_location']}")
    if prefs.get("preferred_locations"):
        lines.append("- Preferred locations for hybrid/on-site work: "
                     + "; ".join(prefs["preferred_locations"]))
    context = ("\nCANDIDATE CONTEXT:\n" + "\n".join(lines) + "\n") if lines else ""
    return SCORE_PROMPT.format(context=context)


def title_excluded(title: str, patterns: list[str]) -> str | None:
    """Return the ``exclude_titles`` pattern a job title matches, if any (case-insensitive)."""
    padded = f" {title.lower()} "
    for pattern in patterns:
        if pattern and pattern.lower() in padded:
            return pattern
    return None


def apply_preferences(result: dict, prefs: dict | None) -> dict:
    """Cap the fit score when the job is the wrong level or in a place the candidate can't work.

    The LLM reports facts (work mode, location match, seniority); the caps are
    applied here so they are predictable. Unknown values never cap.
    """
    prefs = prefs or {}
    caps: list[tuple[int, str]] = []
    # Without a base location the model has nothing to judge location against.
    if prefs.get("base_location") and result.get("location_match") == "no":
        caps.append((3, "location not workable"))
    allowed_modes = prefs.get("work_modes")
    if allowed_modes and result.get("work_mode") in WORK_MODES and result["work_mode"] not in allowed_modes:
        caps.append((3, f"{result['work_mode']} not wanted"))
    allowed_levels = prefs.get("seniority")
    if (allowed_levels and result.get("seniority") in SENIORITY_LEVELS
            and result["seniority"] not in allowed_levels):
        caps.append((4, f"{result['seniority']} level not targeted"))
    min_salary = prefs.get("min_salary")
    if min_salary and result.get("salary_max") and result["salary_max"] < int(min_salary):
        caps.append((5, f"posted max ${result['salary_max']:,} below ${int(min_salary):,} target"))

    if caps:
        cap, reason = min(caps)
        if result["score"] > cap:
            result["reasoning"] = f"[capped {result['score']}->{cap}: {reason}] {result['reasoning']}"
            result["score"] = cap
    return result


def _parse_score_response(response: str) -> dict:
    """Parse the LLM's score response into structured data.

    Args:
        response: Raw LLM response text.

    Returns:
        {"score": int, "keywords": str, "reasoning": str,
         "work_mode": str|None, "location_match": str|None, "seniority": str|None}
    """
    score = 0
    keywords = ""
    reasoning = response
    labels: dict[str, str | None] = {"work_mode": None, "location_match": None, "seniority": None}

    for line in response.split("\n"):
        # Tolerate markdown decoration such as "**SCORE:** 7" or "- SCORE: 7".
        line = re.sub(r"^[\s*#>-]+", "", line).replace("**", "").strip()
        if line.startswith("SCORE:"):
            try:
                score = int(re.search(r"\d+", line).group())
                score = max(1, min(10, score))
            except (AttributeError, ValueError):
                score = 0
        elif line.startswith("KEYWORDS:"):
            keywords = line.replace("KEYWORDS:", "").strip()
        elif line.startswith("REASONING:"):
            reasoning = line.replace("REASONING:", "").strip()
        else:
            for key in labels:
                tag = key.upper() + ":"
                if line.startswith(tag):
                    value = line[len(tag):].strip().strip("[]").lower()
                    labels[key] = None if value in ("", "unknown") else value

    return {"score": score, "keywords": keywords, "reasoning": reasoning, **labels,
            "salary_max": _parse_salary(response)}


def _parse_salary(response: str) -> int | None:
    """Read the SALARY_MAX line as annual USD ("$185,000", "185k", "185000")."""
    # [ \t] not \s: "145000\nKEYWORDS:" must not read the next line's K as "thousand".
    m = re.search(r"SALARY_MAX:[ \t]*\$?[ \t]*([\d,.]+)[ \t]*(k\b)?", response.replace("**", ""),
                  re.IGNORECASE)
    if not m:
        return None
    try:
        value = float(m.group(1).replace(",", ""))
    except ValueError:
        return None
    if m.group(2):
        value *= 1000
    # Below 1000 is a parse slip (e.g. "185" meaning thousands is ambiguous): ignore.
    return int(value) if value >= 1000 else None


def score_job(resume_text: str, job: dict, prefs: dict | None = None) -> dict:
    """Score a single job against the resume.

    Args:
        resume_text: The candidate's full resume text.
        job: Job dict with keys: title, site, location, full_description.
        prefs: The ``preferences`` section of searches.yaml (optional).

    Returns:
        {"score": int, "keywords": str, "reasoning": str, "work_mode": str|None, ...}
        with ``"error": True`` when the job could not be scored.
    """
    job_text = (
        f"TITLE: {job['title']}\n"
        f"COMPANY: {job['site']}\n"
        f"LOCATION: {job.get('location', 'N/A')}\n"
        + (f"SALARY (from the job board): {job['salary']}\n" if job.get("salary") else "")
        + f"\nDESCRIPTION:\n{(job.get('full_description') or '')[:6000]}"
    )

    messages = [
        {"role": "system", "content": build_score_prompt(prefs)},
        {"role": "user", "content": f"RESUME:\n{resume_text}\n\n---\n\nJOB POSTING:\n{job_text}"},
    ]

    try:
        client = get_client("score")
        response = client.chat(messages, max_tokens=512, temperature=0.2)
    except ClaudeUsageLimitError:
        raise
    except Exception as e:
        log.error("LLM error scoring job '%s': %s", job.get("title", "?"), e)
        # Flagged so the job stays unscored and is retried next run.
        return {"score": 0, "keywords": "", "reasoning": f"LLM error: {e}", "error": True}

    result = _parse_score_response(response)
    if result["score"] == 0:
        # Saved as 0 (not retried): a reply that can't be parsed now likely never will be.
        log.error("Unparseable score for '%s': %s", job.get("title", "?"), response[:200])
        return {**result, "reasoning": f"Unparseable LLM reply: {response[:300]}"}
    return apply_preferences(result, prefs)


def run_scoring(limit: int = 0, rescore: bool = False) -> dict:
    """Score unscored jobs that have full descriptions.

    Args:
        limit: Maximum number of jobs to score in this run.
        rescore: If True, re-score all jobs (not just unscored ones).

    Returns:
        {"scored": int, "errors": int, "elapsed": float, "distribution": list}
    """
    resume_text = RESUME_PATH.read_text(encoding="utf-8")
    conn = get_connection()

    if rescore:
        query = "SELECT * FROM jobs WHERE full_description IS NOT NULL"
        if limit > 0:
            query += f" LIMIT {limit}"
        jobs = conn.execute(query).fetchall()
    else:
        jobs = get_jobs_by_stage(conn=conn, stage="pending_score", limit=limit)

    if not jobs:
        log.info("No unscored jobs with descriptions found.")
        return {"scored": 0, "errors": 0, "elapsed": 0.0, "distribution": []}

    # Convert sqlite3.Row to dicts if needed
    if jobs and not isinstance(jobs[0], dict):
        columns = jobs[0].keys()
        jobs = [dict(zip(columns, row)) for row in jobs]

    search_cfg = load_search_config() or {}
    prefs = search_cfg.get("preferences") or {}
    exclude = search_cfg.get("exclude_titles") or []

    log.info("Scoring %d jobs sequentially...", len(jobs))
    t0 = time.time()
    completed = 0
    errors = 0
    results: list[dict] = []

    for job in jobs:
        pattern = title_excluded(job.get("title") or "", exclude)
        try:
            if pattern:
                # No LLM call: the user never wants this title.
                result = {"score": 1, "keywords": "", "work_mode": None,
                          "reasoning": f"Excluded title (exclude_titles: {pattern!r})"}
            else:
                result = score_job(resume_text, job, prefs)
        except ClaudeUsageLimitError as e:
            log.error("Stopping: %s. Unscored jobs stay queued for the next run.", e)
            break
        result["url"] = job["url"]
        completed += 1

        if result.get("error"):
            errors += 1
        else:
            # Saved per job so a long run that stops early keeps its progress.
            conn.execute(
                "UPDATE jobs SET fit_score = ?, score_reasoning = ?, work_mode = ?, scored_at = ? "
                "WHERE url = ?",
                (result["score"], f"{result['keywords']}\n{result['reasoning']}",
                 result.get("work_mode"), datetime.now(timezone.utc).isoformat(), result["url"]),
            )
            conn.commit()
            results.append(result)

        log.info(
            "[%d/%d] score=%d  %s",
            completed, len(jobs), result["score"], job.get("title", "?")[:60],
        )

    elapsed = time.time() - t0
    log.info("Done: %d scored in %.1fs (%.1f jobs/sec)", len(results), elapsed, len(results) / elapsed if elapsed > 0 else 0)

    # Score distribution
    dist = conn.execute("""
        SELECT fit_score, COUNT(*) FROM jobs
        WHERE fit_score IS NOT NULL
        GROUP BY fit_score ORDER BY fit_score DESC
    """).fetchall()
    distribution = [(row[0], row[1]) for row in dist]

    return {
        "scored": len(results),
        "errors": errors,
        "elapsed": elapsed,
        "distribution": distribution,
    }
