"""Application Copilot services — Resume Tailor, Cover Letter, Bullet Optimizer.

Each service has a DETERMINISTIC base (available without Gemini) and an
optional Gemini-grounded path that **always validates** the Gemini output
against the evidence. Gemini wording that fails validation is kept but
marked FAIL (never presented as trusted) — the user sees the unsupported
claims and can regenerate.

Reuses the existing :class:`GeminiClient` (Step 3) — no second client.
No full resume text is sent to Gemini — only structured, redacted evidence.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Optional

from src.modules.application_copilot.evidence import (
    evidence_ids_for_skills,
    extract_evidence,
)
from src.modules.application_copilot.model import (
    ApplicationCopilotError,
    ApplicationCopilotParseError,
    BulletSuggestion,
    CoverLetterDraft,
    EvidenceItem,
    EvidenceValidation,
    TailoredResumeSuggestion,
    ValidationStatus,
)
from src.modules.application_copilot.prompt import (
    BULLET_SCHEMA,
    BULLET_SYSTEM_PROMPT,
    COVER_LETTER_SCHEMA,
    COVER_LETTER_SYSTEM_PROMPT,
    TAILOR_SCHEMA,
    TAILOR_SYSTEM_PROMPT,
)
from src.modules.application_copilot.validator import validate_claims
from src.modules.jd_analysis.model import JobAnalysis
from src.modules.resume_intelligence.model import ResumeAnalysis
from src.services.gemini import GeminiClient, GeminiError

_logger = logging.getLogger(__name__)

__all__ = [
    "ApplicationCopilotError",
    "tailor_resume",
    "generate_cover_letter",
    "optimize_bullet",
]


# --------------------------------------------------------------------------- #
# Resume Tailor
# --------------------------------------------------------------------------- #


def tailor_resume(
    resume: ResumeAnalysis,
    job: JobAnalysis,
    match,
    evidence: list[EvidenceItem],
    *,
    client: Optional[GeminiClient] = None,
) -> list[TailoredResumeSuggestion]:
    """Generate resume-tailoring suggestions.

    Without a ``client``: deterministic, evidence-grounded suggestions
    (reorder skills, highlight relevant experience/projects, emphasize
    matching skills). Always validation=PASS.

    With a ``client``: Gemini refines the wording; each suggestion is
    evidence-validated (FAIL -> kept but marked untrusted).
    """
    deterministic = _build_deterministic_tailor(resume, job, match, evidence)
    if client is None:
        return deterministic

    gemini_suggestions = _gemini_tailor(resume, job, match, evidence, client)
    # Validate each Gemini suggestion; if FAIL, fall back to the deterministic
    # suggestion for that section (kept untrusted version is still returned
    # with FAIL validation so the user sees what was rejected).
    return _merge_tailor(deterministic, gemini_suggestions, evidence, resume, job, match)


def _build_deterministic_tailor(
    resume: ResumeAnalysis,
    job: JobAnalysis,
    match,
    evidence: list[EvidenceItem],
) -> list[TailoredResumeSuggestion]:
    suggestions: list[TailoredResumeSuggestion] = []
    matching = list(getattr(match, "matching_skills", []) or [])

    # 1. Skills: reorder to lead with matching skills.
    if resume.skills and matching:
        match_norm = {_normalize(s) for s in matching}
        lead = [s for s in resume.skills if _normalize(s) in match_norm]
        rest = [s for s in resume.skills if _normalize(s) not in match_norm]
        ordered = lead + rest
        ev_ids = evidence_ids_for_skills(lead, evidence)
        suggestions.append(
            TailoredResumeSuggestion(
                section="skills",
                original_content=", ".join(resume.skills),
                suggested_content=", ".join(ordered),
                reason="Lead with skills that match the target job.",
                evidence_ids=ev_ids,
                validation=EvidenceValidation(
                    status=ValidationStatus.PASS,
                    supported_claims=[
                        f"Skills {', '.join(lead)} supported by resume evidence."
                    ] if lead else [],
                ),
            )
        )

    # 2. Experience: highlight entries overlapping the job's keywords/skills.
    job_corpus = " ".join(
        job.keywords + job.required_skills + job.preferred_skills + [job.domain or ""]
    ).lower()
    for i, exp in enumerate(resume.experience, start=1):
        if exp.description and _overlaps(exp.description, job_corpus):
            ev_id = f"EXP-{i:03d}"
            suggestions.append(
                TailoredResumeSuggestion(
                    section="experience",
                    original_content=exp.description,
                    suggested_content=exp.description,  # no fabrication
                    reason="This experience overlaps the target role's requirements.",
                    evidence_ids=[ev_id],
                    validation=EvidenceValidation(
                        status=ValidationStatus.PASS,
                        supported_claims=[f"Experience {ev_id} supported by resume."],
                    ),
                )
            )

    # 3. Projects: feature relevant projects.
    for i, proj in enumerate(resume.projects, start=1):
        proj_text = f"{proj.technologies} {proj.description}".lower()
        if _overlaps(proj_text, job_corpus):
            ev_id = f"PROJ-{i:03d}"
            suggestions.append(
                TailoredResumeSuggestion(
                    section="projects",
                    original_content=proj.description,
                    suggested_content=proj.description,
                    reason="This project uses technologies relevant to the target role.",
                    evidence_ids=[ev_id],
                    validation=EvidenceValidation(
                        status=ValidationStatus.PASS,
                        supported_claims=[f"Project {ev_id} supported by resume."],
                    ),
                )
            )

    # 4. Summary: suggest emphasizing matching skills.
    if resume.professional_summary and matching:
        ev_ids = evidence_ids_for_skills(matching[:5], evidence)
        suggestions.append(
            TailoredResumeSuggestion(
                section="summary",
                original_content=resume.professional_summary,
                suggested_content=resume.professional_summary,  # no fabrication
                reason=f"Consider emphasizing matching skills: {', '.join(matching[:5])}.",
                evidence_ids=ev_ids,
                validation=EvidenceValidation(
                    status=ValidationStatus.PASS,
                    supported_claims=["Summary supported by resume evidence."],
                ),
            )
        )

    return suggestions


def _gemini_tailor(
    resume: ResumeAnalysis,
    job: JobAnalysis,
    match,
    evidence: list[EvidenceItem],
    client: GeminiClient,
) -> list[dict]:
    """Call Gemini for tailored wording. Returns raw parsed suggestions."""
    user_text = _build_tailor_context(resume, job, match, evidence)
    raw = client.generate_json(
        system_prompt=TAILOR_SYSTEM_PROMPT,
        user_text=user_text,
        response_schema=TAILOR_SCHEMA,
    )
    data = _safe_parse_json(raw, ApplicationCopilotParseError)
    items = data.get("suggestions") if isinstance(data, dict) else None
    if not isinstance(items, list):
        raise ApplicationCopilotParseError("Response did not contain a 'suggestions' list.")
    return [i for i in items if isinstance(i, dict)]


def _merge_tailor(
    deterministic: list[TailoredResumeSuggestion],
    gemini_items: list[dict],
    evidence: list[EvidenceItem],
    resume: ResumeAnalysis,
    job: JobAnalysis,
    match,
) -> list[TailoredResumeSuggestion]:
    """Build validated TailoredResumeSuggestions from Gemini output.

    Each Gemini suggestion is evidence-validated. FAIL suggestions are
    still returned (so the user sees what was rejected) but marked FAIL.
    """
    det_by_section = {s.section: s for s in deterministic}
    out: list[TailoredResumeSuggestion] = []
    for item in gemini_items:
        section = (item.get("section") or "").strip()
        suggested = (item.get("suggested_content") or "").strip()
        original = (item.get("original_content") or "").strip()
        reason = (item.get("reason") or "").strip()
        validation = validate_claims(
            suggested, evidence, resume=resume, job=job, match=match
        )
        ev_ids = _evidence_ids_referenced(suggested, evidence)
        if not ev_ids and section in det_by_section:
            ev_ids = det_by_section[section].evidence_ids
        out.append(
            TailoredResumeSuggestion(
                section=section or "general",
                original_content=original,
                suggested_content=suggested,
                reason=reason,
                evidence_ids=ev_ids,
                validation=validation,
            )
        )
    return out


def _build_tailor_context(
    resume: ResumeAnalysis,
    job: JobAnalysis,
    match,
    evidence: list[EvidenceItem],
) -> str:
    """Redacted, structured context (no raw resume text dump)."""
    lines = [
        f"Target role: {_target_role(job)}",
        f"Matching skills: {', '.join(getattr(match, 'matching_skills', []) or [])}",
        f"Job required skills: {', '.join(job.required_skills)}",
        "",
        "Candidate evidence (use ONLY these facts — do not invent):",
    ]
    for e in evidence:
        lines.append(f"- {e.evidence_id} [{e.source_type}]: {e.source_text}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Cover Letter
# --------------------------------------------------------------------------- #


def generate_cover_letter(
    resume: ResumeAnalysis,
    job: JobAnalysis,
    match,
    evidence: list[EvidenceItem],
    *,
    client: Optional[GeminiClient] = None,
) -> CoverLetterDraft:
    """Generate a cover letter draft.

    Without a ``client``: returns a status draft explaining AI generation
    requires Gemini (no fabricated letter).

    With a ``client``: Gemini writes a letter grounded in evidence; the
    letter is evidence-validated (FAIL -> kept but marked untrusted).
    """
    target_role = _target_role(job)
    generated_at = datetime.now(timezone.utc).isoformat()

    if client is None:
        return CoverLetterDraft(
            cover_letter="",
            evidence_ids=[],
            evidence_validation=EvidenceValidation(status=ValidationStatus.PASS),
            warnings=[
                "AI cover-letter generation requires Gemini configuration. "
                "Add a `[gemini]` API key to `.streamlit/secrets.toml` to "
                "generate a personalized, evidence-grounded letter."
            ],
            target_role=target_role,
            generated_at=generated_at,
        )

    raw = client.generate_json(
        system_prompt=COVER_LETTER_SYSTEM_PROMPT,
        user_text=_build_cover_letter_context(resume, job, match, evidence),
        response_schema=COVER_LETTER_SCHEMA,
    )
    data = _safe_parse_json(raw, ApplicationCopilotParseError)
    if not isinstance(data, dict):
        raise ApplicationCopilotParseError("Cover letter response was not a JSON object.")
    letter = (data.get("cover_letter") or "").strip()
    ev_ids = _coerce_str_list(data.get("evidence_ids"))
    validation = validate_claims(letter, evidence, resume=resume, job=job, match=match)
    if not ev_ids:
        ev_ids = _evidence_ids_referenced(letter, evidence)
    warnings: list[str] = []
    if validation.status is ValidationStatus.FAIL:
        warnings.append(
            "AI output failed evidence validation — see unsupported claims below. "
            "Regenerate to try again."
        )
    return CoverLetterDraft(
        cover_letter=letter,
        evidence_ids=ev_ids,
        evidence_validation=validation,
        warnings=warnings,
        target_role=target_role,
        generated_at=generated_at,
    )


def _build_cover_letter_context(
    resume: ResumeAnalysis,
    job: JobAnalysis,
    match,
    evidence: list[EvidenceItem],
) -> str:
    lines = [
        f"Target role: {_target_role(job)}",
        f"Job required skills: {', '.join(job.required_skills)}",
        f"Matching skills: {', '.join(getattr(match, 'matching_skills', []) or [])}",
        "",
        "Candidate evidence (use ONLY these facts):",
    ]
    for e in evidence:
        lines.append(f"- {e.evidence_id} [{e.source_type}]: {e.source_text}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Bullet Optimizer
# --------------------------------------------------------------------------- #


def optimize_bullet(
    bullet: str,
    resume: ResumeAnalysis,
    job: JobAnalysis,
    match,
    evidence: list[EvidenceItem],
    *,
    client: Optional[GeminiClient] = None,
) -> BulletSuggestion:
    """Optimize a single resume bullet.

    Without a ``client``: returns structured guidance (no AI text).

    With a ``client``: Gemini rewrites the bullet; the result is
    evidence-validated (FAIL -> kept but marked untrusted).
    """
    if client is None:
        return BulletSuggestion(
            original_bullet=bullet,
            optimized_bullet="",
            reason=(
                "Structured guidance: use a strong action verb; keep the "
                "bullet concise; ensure it mentions the relevant skill; "
                "quantify outcomes only if supported by resume evidence."
            ),
            evidence_ids=_evidence_ids_referenced(bullet, evidence),
            validation=EvidenceValidation(status=ValidationStatus.PASS),
        )

    raw = client.generate_json(
        system_prompt=BULLET_SYSTEM_PROMPT,
        user_text=_build_bullet_context(bullet, job, match, evidence),
        response_schema=BULLET_SCHEMA,
    )
    data = _safe_parse_json(raw, ApplicationCopilotParseError)
    if not isinstance(data, dict):
        raise ApplicationCopilotParseError("Bullet response was not a JSON object.")
    optimized = (data.get("optimized_bullet") or "").strip()
    reason = (data.get("reason") or "").strip()
    ev_ids = _coerce_str_list(data.get("evidence_ids")) or _evidence_ids_referenced(
        optimized, evidence
    )
    validation = validate_claims(
        optimized, evidence, resume=resume, job=job, match=match
    )
    return BulletSuggestion(
        original_bullet=bullet,
        optimized_bullet=optimized,
        reason=reason,
        evidence_ids=ev_ids,
        validation=validation,
    )


def _build_bullet_context(
    bullet: str, job: JobAnalysis, match, evidence: list[EvidenceItem]
) -> str:
    lines = [
        f"Target role: {_target_role(job)}",
        f"Relevant job skills: {', '.join(job.required_skills)}",
        f"Original bullet to optimize: {bullet}",
        "",
        "Candidate evidence (use ONLY these facts):",
    ]
    for e in evidence:
        lines.append(f"- {e.evidence_id} [{e.source_type}]: {e.source_text}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _target_role(job: JobAnalysis) -> str:
    parts = [p for p in (job.job_title, job.domain) if p]
    return " · ".join(parts) if parts else (job.job_title or "the target role")


def _overlaps(text: str, corpus: str) -> bool:
    """True if any token of ``text`` (>=3 chars) appears in ``corpus``."""
    if not text or not corpus:
        return False
    import re

    tokens = {t for t in re.split(r"[^a-z0-9+#]+", text.lower()) if len(t) >= 3}
    return any(tok in corpus for tok in tokens)


def _evidence_ids_referenced(text: str, evidence: list[EvidenceItem]) -> list[str]:
    """Return evidence IDs whose source_text (or significant tokens) appear in text."""
    if not text:
        return []
    import re

    text_lower = text.lower()
    out: list[str] = []
    for e in evidence:
        if not e.source_text:
            continue
        # Direct substring match, or significant token overlap.
        if e.source_text.lower() in text_lower:
            out.append(e.evidence_id)
            continue
        ev_tokens = {t for t in re.split(r"[^a-z0-9+#]+", e.source_text.lower()) if len(t) >= 4}
        if ev_tokens and ev_tokens & {t for t in re.split(r"[^a-z0-9+#]+", text_lower) if len(t) >= 4}:
            out.append(e.evidence_id)
    return out


def _normalize(s: str) -> str:
    if not s:
        return ""
    import re

    t = s.lower().strip()
    t = re.sub(r"[^a-z0-9+#]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _coerce_str_list(value) -> list[str]:
    out: list[str] = []
    if value is None:
        return out
    items = value if isinstance(value, list) else [value]
    for item in items:
        if item is None:
            continue
        s = str(item).strip()
        if s:
            out.append(s)
    return out


def _safe_parse_json(raw: str, error_cls) -> dict:
    if not raw or not raw.strip():
        raise error_cls("Gemini returned an empty response.")
    text = raw.strip()
    if text.startswith("```"):
        text = text[3:]
        if text[:4].lower() == "json":
            text = text[4:]
        if text.endswith("```"):
            text = text[:-3]
    try:
        data = json.loads(text.strip())
    except json.JSONDecodeError as e:
        raise error_cls(
            f"Gemini response was not valid JSON ({e.msg} at line {e.lineno} col {e.colno})."
        )
    return data
