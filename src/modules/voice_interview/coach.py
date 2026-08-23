"""Interview Coach services — question generation + answer evaluation.

Each service has a DETERMINISTIC base (no Gemini) and an optional
Gemini-grounded path that ALWAYS validates output.

- :func:`generate_questions` — deterministic fallback questions, optionally
  refined by Gemini. Every Gemini question is validated via
  :func:`validate_question` (reuses the Application Copilot evidence
  architecture); invalid questions are replaced with the deterministic
  fallback for that slot.
- :func:`evaluate_answer` — deterministic heuristic evaluation, optionally
  enriched by Gemini. Unsupported candidate claims are detected
  deterministically (authoritative — Gemini cannot suppress them).
- :func:`build_report` — deterministic final readiness report.

Reuses the existing :class:`GeminiClient` (Step 3) — no second client.
No raw ``resume_text`` is sent to Gemini — only structured evidence.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone
from typing import Optional

from src.modules.application_copilot.evidence import (
    EvidenceItem,
    evidence_corpus,
    extract_evidence,
)
from src.modules.application_copilot.validator import validate_claims
from src.modules.voice_interview.model import (
    AnswerEvaluation,
    Difficulty,
    InterviewAnswer,
    InterviewCategory,
    InterviewQuestion,
    InterviewReport,
    clamp_score,
)
from src.modules.voice_interview.planner import build_questions, plan_distribution
from src.modules.voice_interview.prompt import (
    ANSWER_EVAL_SCHEMA,
    ANSWER_EVAL_SYSTEM_PROMPT,
    QUESTION_GEN_SCHEMA,
    QUESTION_GEN_SYSTEM_PROMPT,
)
from src.modules.voice_interview.validator import validate_question
from src.services.gemini import GeminiClient, GeminiError

_logger = logging.getLogger(__name__)

__all__ = [
    "generate_questions",
    "evaluate_answer",
    "build_report",
    "detect_unsupported_answer_claims",
]


# --------------------------------------------------------------------------- #
# Question generation
# --------------------------------------------------------------------------- #


def generate_questions(
    resume,
    job,
    match,
    skill_gap_report,
    *,
    n: int = 10,
    difficulty: Difficulty = Difficulty.MEDIUM,
    client: Optional[GeminiClient] = None,
    evidence: Optional[list[EvidenceItem]] = None,
) -> list[InterviewQuestion]:
    """Generate ``n`` interview questions.

    Deterministic fallback questions are always built. If a ``client`` is
    provided, Gemini may refine wording; every Gemini question is validated
    and invalid ones are replaced with the deterministic fallback.
    """
    if evidence is None and resume is not None:
        evidence = extract_evidence(resume)

    distribution = plan_distribution(n)
    deterministic = build_questions(
        distribution, resume, job, match, skill_gap_report, difficulty=difficulty
    )

    if client is None:
        return deterministic

    # Gemini path: ask for refined wording per slot.
    try:
        raw = client.generate_json(
            system_prompt=QUESTION_GEN_SYSTEM_PROMPT,
            user_text=_build_question_context(resume, job, match, skill_gap_report, deterministic),
            response_schema=QUESTION_GEN_SCHEMA,
        )
        gemini_qs = _parse_gemini_questions(raw)
    except GeminiError:
        raise
    except Exception:
        _logger.exception("Unexpected error during Gemini question generation")
        return deterministic

    # Merge: validate each Gemini question; invalid -> keep deterministic.
    out: list[InterviewQuestion] = []
    for i in range(len(deterministic)):
        det = deterministic[i] if i < len(deterministic) else None
        gem = gemini_qs[i] if i < len(gemini_qs) else None
        if gem is not None:
            is_valid, _reasons = validate_question(
                gem.get("question", ""), evidence or [], resume, job, match,
                skill_gap_report,
            )
            if is_valid:
                out.append(_merge_question(det, gem))
                continue
        if det is not None:
            out.append(det)
        else:
            # Fallback to a generic behavioral question if both missing.
            out.append(_fallback_behavioral(f"IQ-{i+1:03d}", difficulty))
    return out


def _merge_question(det: Optional[InterviewQuestion], gem: dict) -> InterviewQuestion:
    """Construct the final question: deterministic metadata + Gemini wording ONLY.

    The deterministic planner is AUTHORITATIVE over all structural metadata
    (question_id, category, difficulty, target_skill, source,
    related_job_requirement, expected_evidence). Gemini may ONLY improve the
    question wording and (optionally) the rationale. Gemini's
    category/difficulty/target_skill/source/related_job_requirement are
    deliberately IGNORED so it cannot redefine the interview plan.
    """
    if det is None:
        # No deterministic slot — should not happen, but handle safely.
        return _fallback_behavioral("", Difficulty.MEDIUM)
    gem_question = (gem.get("question") or "").strip()
    gem_rationale = (gem.get("rationale") or "").strip()
    return InterviewQuestion(
        question_id=det.question_id,  # DETERMINISTIC
        question=gem_question or det.question,  # Gemini wording, fallback to deterministic
        category=det.category,  # DETERMINISTIC — Gemini's category IGNORED
        difficulty=det.difficulty,  # DETERMINISTIC — Gemini's difficulty IGNORED
        target_skill=det.target_skill,  # DETERMINISTIC — Gemini's target_skill IGNORED
        source=det.source,  # DETERMINISTIC — Gemini's source IGNORED
        rationale=gem_rationale or det.rationale,  # Gemini rationale if provided, else deterministic
        expected_evidence=det.expected_evidence,  # DETERMINISTIC
        related_job_requirement=det.related_job_requirement,  # DETERMINISTIC
    )


def _parse_gemini_questions(raw: str) -> list[dict]:
    if not raw or not raw.strip():
        return []
    text = _strip_fence(raw).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    if not isinstance(data, dict):
        return []
    qs = data.get("questions")
    if not isinstance(qs, list):
        return []
    return [q for q in qs if isinstance(q, dict)]


def _build_question_context(resume, job, match, skill_gap_report, deterministic):
    """Redacted, structured context for Gemini (no raw resume text)."""
    lines = ["Target role: " + _target_role(job)]
    if resume is not None:
        lines.append(f"Resume skills: {', '.join(getattr(resume, 'skills', []) or [])}")
        if getattr(resume, "projects", None):
            lines.append("Projects: " + ", ".join(p.name for p in resume.projects))
        if getattr(resume, "experience", None):
            lines.append("Experience employers: " + ", ".join(e.company for e in resume.experience if e.company))
    if job is not None:
        lines.append(f"Job required skills: {', '.join(getattr(job, 'required_skills', []) or [])}")
        lines.append(f"Job preferred skills: {', '.join(getattr(job, 'preferred_skills', []) or [])}")
    if match is not None:
        lines.append(f"Matching skills: {', '.join(getattr(match, 'matching_skills', []) or [])}")
        lines.append(f"Missing skills: {', '.join(getattr(match, 'missing_skills', []) or [])}")
    if skill_gap_report is not None and getattr(skill_gap_report, "skill_gaps", None):
        lines.append("Skill gaps: " + ", ".join(g.skill for g in skill_gap_report.skill_gaps))
    lines.append("")
    lines.append("Question plan (refine wording ONLY — do not invent facts; do not claim the candidate has a missing skill):")
    for q in deterministic:
        lines.append(f"- [{q.category.value}] {q.question}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Answer evaluation
# --------------------------------------------------------------------------- #


def evaluate_answer(
    question: InterviewQuestion,
    answer_text: str,
    resume,
    job,
    match,
    *,
    client: Optional[GeminiClient] = None,
    evidence: Optional[list[EvidenceItem]] = None,
) -> AnswerEvaluation:
    """Evaluate a candidate's answer.

    Unsupported candidate claims are detected DETERMINISTICALLY
    (authoritative — Gemini cannot suppress them). Scores come from the
    deterministic heuristic, optionally enriched by Gemini.
    """
    if evidence is None and resume is not None:
        evidence = extract_evidence(resume)

    unsupported = detect_unsupported_answer_claims(
        answer_text, evidence or [], resume, job, match
    )
    evidence_alignment = (
        "unsupported" if unsupported else ("aligned" if answer_text.strip() else "partial")
    )

    if client is None:
        return _deterministic_evaluation(
            question, answer_text, unsupported, evidence_alignment
        )

    # Gemini path.
    try:
        raw = client.generate_json(
            system_prompt=ANSWER_EVAL_SYSTEM_PROMPT,
            user_text=_build_eval_context(question, answer_text, resume, job, match),
            response_schema=ANSWER_EVAL_SCHEMA,
        )
        parsed = _safe_parse_eval(raw)
    except GeminiError:
        raise
    except Exception:
        _logger.exception("Unexpected error during Gemini answer evaluation")
        return _deterministic_evaluation(
            question, answer_text, unsupported, evidence_alignment
        )

    # Merge Gemini scores + DETERMINISTIC unsupported claims (authoritative).
    parsed["question_id"] = question.question_id
    parsed["unsupported_claims"] = unsupported or parsed.get("unsupported_claims", [])
    parsed["evidence_alignment"] = evidence_alignment if unsupported else parsed.get("evidence_alignment", "aligned")
    # Clamp all scores to 0-10.
    for k in ("relevance_score", "technical_score", "specificity_score",
              "communication_score", "completeness_score", "overall_score"):
        if k in parsed:
            parsed[k] = clamp_score(parsed[k])
    return AnswerEvaluation.from_dict(parsed)


def detect_unsupported_answer_claims(
    answer_text: str,
    evidence: list[EvidenceItem],
    resume,
    job,
    match,
) -> list[str]:
    """Detect candidate claims not supported by resume evidence.

    Reuses the Application Copilot :func:`validate_claims` (metrics, years,
    missing-skills-claimed, certs, proper nouns) + an extended count-claim
    check (servers/clusters/etc.) + an acronym/technology check for
    interview answers.
    """
    if not answer_text or not answer_text.strip():
        return []
    v = validate_claims(answer_text, evidence, resume=resume, job=job, match=match)
    claims = list(v.unsupported_claims)
    # In an answer context, a proper-noun warning IS an unsupported claim.
    for w in v.warnings:
        claims.append(w)
    corpus = evidence_corpus(evidence)
    # Extended count-claim check (broader nouns than the Copilot validator).
    for m in _extract_extended_metrics(answer_text):
        if m.lower() not in corpus:
            claims.append(f"Specific claim '{m}' not found in resume evidence.")
    # Acronym/technology check: all-caps tokens (AWS, K8s, GCP) not in
    # evidence and not a job required/preferred skill -> unsupported.
    job_skills = set()
    if job is not None:
        for s in list(getattr(job, "required_skills", []) or []) + list(
            getattr(job, "preferred_skills", []) or []
        ):
            job_skills.add(_normalize_skill(s))
    for acr in _extract_acronyms(answer_text):
        key = acr.lower()
        if key in corpus or _normalize_skill(acr) in job_skills:
            continue
        claims.append(
            f"Technology '{acr}' mentioned in the answer is not found in "
            f"resume evidence."
        )
    # Dedupe preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for c in claims:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _deterministic_evaluation(
    question: InterviewQuestion,
    answer_text: str,
    unsupported: list[str],
    evidence_alignment: str,
) -> AnswerEvaluation:
    """Heuristic 0-10 evaluation without Gemini (no fabricated correctness)."""
    text = (answer_text or "").strip()
    words = len(text.split())
    sentences = max(1, len(re.findall(r"[.!?]+", text)) + (1 if text else 0))

    # Relevance: does the answer mention the target skill / job requirement?
    target = (question.target_skill or question.related_job_requirement or "").lower()
    relevance = 5.0
    if target and target in text.lower():
        relevance = 7.0
    if words == 0:
        relevance = 0.0

    # Specificity: concrete details (numbers, evidence terms).
    has_numbers = bool(re.search(r"\d", text))
    specificity = 6.0 if has_numbers else (4.0 if words > 20 else 2.0)

    # Communication: length + sentence structure.
    communication = clamp_score(min(10.0, words / 10.0)) if words else 0.0

    # Completeness: length-based.
    completeness = clamp_score(min(10.0, words / 15.0)) if words else 0.0

    # Technical: conservative — do NOT claim correctness without Gemini.
    technical = 5.0 if (target and target in text.lower()) else 3.0
    if unsupported:
        technical = min(technical, 4.0)

    overall = clamp_score(
        0.30 * relevance + 0.25 * technical + 0.15 * specificity
        + 0.15 * communication + 0.15 * completeness
    )

    strengths: list[str] = []
    improvements: list[str] = []
    if relevance >= 7:
        strengths.append("Answer addresses the core of the question.")
    if specificity >= 6:
        strengths.append("Includes concrete specifics.")
    if words < 30:
        improvements.append("Expand the answer with more detail and a concrete example.")
    if not has_numbers and question.category is InterviewCategory.TECHNICAL:
        improvements.append("Add measurable outcomes or specific technical details where truthful.")
    if unsupported:
        improvements.append(
            "Some claims are not supported by the resume evidence — be "
            "prepared to substantiate them."
        )

    recommendation = (
        "Practice articulating your experience with concrete, evidence-backed "
        "examples. AI evaluation (Gemini) can provide deeper technical feedback."
    )
    if unsupported:
        recommendation = (
            "Your answer contains specific claims not supported by the available "
            "resume evidence. If this reflects real experience, be prepared to "
            "substantiate it. AI evaluation (Gemini) can provide deeper feedback."
        )

    return AnswerEvaluation(
        question_id=question.question_id,
        relevance_score=relevance,
        technical_score=technical,
        specificity_score=specificity,
        communication_score=communication,
        completeness_score=completeness,
        overall_score=overall,
        strengths=strengths,
        improvements=improvements,
        missing_points=[] if words > 30 else ["Consider adding a concrete example."],
        evidence_alignment=evidence_alignment,
        unsupported_claims=unsupported,
        recommendation=recommendation,
    )


def _build_eval_context(question, answer_text, resume, job, match):
    lines = [
        f"Target role: {_target_role(job)}",
        f"Question ({question.category.value}/{question.difficulty.value}): {question.question}",
        f"Target skill: {question.target_skill or '(none)'}",
        f"Related job requirement: {question.related_job_requirement or '(none)'}",
        f"Candidate answer: {answer_text}",
    ]
    if resume is not None:
        lines.append(f"Resume skills: {', '.join(getattr(resume, 'skills', []) or [])}")
        if getattr(resume, "projects", None):
            lines.append("Projects: " + ", ".join(p.name for p in resume.projects))
    if match is not None:
        lines.append(f"Missing skills: {', '.join(getattr(match, 'missing_skills', []) or [])}")
    return "\n".join(lines)


def _safe_parse_eval(raw: str) -> dict:
    if not raw or not raw.strip():
        return {}
    text = _strip_fence(raw).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


# --------------------------------------------------------------------------- #
# Final report
# --------------------------------------------------------------------------- #


def build_report(
    questions: list[InterviewQuestion],
    evaluations: dict[str, AnswerEvaluation],
    target_role: str = "",
) -> InterviewReport:
    """Deterministic final readiness report.

    Readiness formula (documented, heuristic — NOT a hiring prediction):
        readiness = round(10 * (
            0.30 * technical_avg +
            0.20 * behavioral_avg +
            0.20 * communication_avg +
            0.20 * job_alignment_avg +
            0.10 * completeness_avg
        ))
    All per-evaluation scores are 0-10; readiness is 0-100. Clamped.
    """
    if not evaluations:
        return InterviewReport(
            readiness_score=0.0,
            total_questions=len(questions),
            completed_questions=0,
            strengths=[],
            weaknesses=["No answers evaluated yet."],
            recommended_practice=["Complete the interview to receive a readiness report."],
        )

    by_qid = {q.question_id: q for q in questions}
    tech_scores: list[float] = []
    beh_scores: list[float] = []
    comm_scores: list[float] = []
    job_scores: list[float] = []
    comp_scores: list[float] = []
    overall_scores: list[float] = []

    for qid, ev in evaluations.items():
        q = by_qid.get(qid)
        tech_scores.append(ev.technical_score)
        comm_scores.append(ev.communication_score)
        comp_scores.append(ev.completeness_score)
        overall_scores.append(ev.overall_score)
        if q is not None and q.category is InterviewCategory.BEHAVIORAL:
            beh_scores.append(ev.overall_score)
        if q is not None and q.category in (
            InterviewCategory.JOB_SPECIFIC, InterviewCategory.SKILL_GAP
        ):
            job_scores.append(ev.relevance_score)

    def _avg(xs):
        return sum(xs) / len(xs) if xs else 0.0

    technical_avg = _avg(tech_scores)
    behavioral_avg = _avg(beh_scores) if beh_scores else _avg(overall_scores)
    communication_avg = _avg(comm_scores)
    job_alignment_avg = _avg(job_scores) if job_scores else _avg(overall_scores)
    completeness_avg = _avg(comp_scores)

    readiness = 10.0 * (
        0.30 * technical_avg
        + 0.20 * behavioral_avg
        + 0.20 * communication_avg
        + 0.20 * job_alignment_avg
        + 0.10 * completeness_avg
    )
    readiness = max(0.0, min(100.0, round(readiness)))

    strengths: list[str] = []
    weaknesses: list[str] = []
    if technical_avg >= 7:
        strengths.append(f"Technical responses averaged {technical_avg:.1f}/10.")
    if communication_avg >= 7:
        strengths.append(f"Communication averaged {communication_avg:.1f}/10.")
    if job_alignment_avg < 5:
        weaknesses.append(f"Job-alignment averaged only {job_alignment_avg:.1f}/10 — practice role-specific questions.")
    if technical_avg < 5:
        weaknesses.append(f"Technical depth averaged {technical_avg:.1f}/10 — deepen core-skill explanations.")
    unsupported_count = sum(1 for ev in evaluations.values() if ev.unsupported_claims)
    if unsupported_count:
        weaknesses.append(f"{unsupported_count} answer(s) contained unsupported claims — ground your responses in real experience.")

    recommended = [
        "Rehearse answers aloud with concrete, evidence-backed examples.",
        "Prepare a 60-second overview of each resume project.",
        "For known skill gaps, articulate an honest learning plan.",
    ]
    if technical_avg < 7:
        recommended.append("Deepen technical explanations for your strongest skills.")
    if job_alignment_avg < 7:
        recommended.append("Practice job-specific and skill-gap questions for the target role.")

    return InterviewReport(
        readiness_score=readiness,
        technical_score=round(technical_avg, 1),
        behavioral_score=round(behavioral_avg, 1),
        communication_score=round(communication_avg, 1),
        job_alignment_score=round(job_alignment_avg, 1),
        strengths=strengths or ["Completed the interview practice session."],
        weaknesses=weaknesses or ["No major weaknesses identified — keep practicing."],
        recommended_practice=recommended,
        completed_questions=len(evaluations),
        total_questions=len(questions),
    )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _target_role(job) -> str:
    if job is None:
        return "the target role"
    parts = [p for p in (getattr(job, "job_title", ""), getattr(job, "domain", "")) if p]
    return " · ".join(parts) if parts else (getattr(job, "job_title", "") or "the target role")


def _fallback_behavioral(qid: str, difficulty: Difficulty) -> InterviewQuestion:
    return InterviewQuestion(
        question_id=qid,
        question="Tell me about a time you faced a significant technical challenge and how you resolved it.",
        category=InterviewCategory.BEHAVIORAL,
        difficulty=difficulty,
        source="GENERAL",
        rationale="General behavioral fallback.",
    )


def _strip_fence(text: str) -> str:
    t = text.strip()
    if t.startswith("```"):
        t = t[3:]
        if t[:4].lower() == "json":
            t = t[4:]
        if t.endswith("```"):
            t = t[:-3]
    return t


# Extended count-claim nouns for interview answers (broader than the
# Application Copilot validator's set, to catch "500 servers", "10 clusters").
_EXTENDED_COUNT_NOUNS = (
    "servers", "clusters", "nodes", "instances", "requests", "users",
    "queries", "deployments", "services", "microservices", "applications",
    "systems", "teams", "machines", "containers", "shards", "partitions",
    "engineers", "people", "members", "reports", "developers", "directs",
)


def _extract_extended_metrics(text: str) -> list[str]:
    """Extract percentage / multiplier / large-number / count-claim strings.

    Also catches plain 2+-digit numbers (e.g. 500) as potential metrics.
    """
    metrics: list[str] = []
    for m in re.finditer(r"\d+(?:\.\d+)?\s*%", text):
        metrics.append(m.group(0).replace(" ", ""))
    for m in re.finditer(r"\b\d+\s*[xX]\b", text):
        metrics.append(m.group(0).replace(" ", "").lower())
    for m in re.finditer(r"\b\d+(?:\.\d+)?\s*[kKmM]\b", text):
        metrics.append(m.group(0).replace(" ", "").lower())
    for m in re.finditer(
        r"\b\d+\s+(?:" + "|".join(_EXTENDED_COUNT_NOUNS) + r")\b",
        text,
        re.IGNORECASE,
    ):
        metrics.append(m.group(0).lower())
    # Plain 2+-digit numbers (e.g. 500, 1000) as potential metrics.
    for m in re.finditer(r"\b\d{2,}\b", text):
        metrics.append(m.group(0))
    seen: set[str] = set()
    out: list[str] = []
    for m in metrics:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def _extract_acronyms(text: str) -> list[str]:
    """Extract all-caps tokens of length >= 2 (AWS, K8s, GCP, SQL)."""
    out: list[str] = []
    seen: set[str] = set()
    for m in re.finditer(r"\b[A-Z][A-Z0-9]{1,}\b", text):
        tok = m.group(0)
        key = tok.lower()
        if key not in seen:
            seen.add(key)
            out.append(tok)
    return out


def _normalize_skill(s: str) -> str:
    if not s:
        return ""
    t = s.lower().strip()
    t = re.sub(r"[^a-z0-9+#]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t
