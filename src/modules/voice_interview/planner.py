"""Deterministic interview question planner.

Builds a question plan from the candidate's ResumeAnalysis, JobAnalysis,
MatchAnalysis, and SkillGapReport — **without Gemini**. The plan assigns a
category distribution and fills each slot with a grounded fallback
question template that references ONLY real resume/job/skill-gap data.

No candidate facts are invented. If a category has no grounding data
(e.g. no projects for the PROJECT category), the slot is reassigned to a
category that does have data.

Category distribution (configurable; default for 10 questions):
    3 technical · 2 job_specific · 2 resume/project · 2 behavioral · 1 skill_gap
The distribution scales proportionally with the total question count.
"""

from __future__ import annotations

from typing import Optional

from src.modules.voice_interview.model import (
    Difficulty,
    InterviewCategory,
    InterviewQuestion,
)

# Default category ratios (sum to 1.0). Tunable.
DEFAULT_RATIOS: dict[InterviewCategory, float] = {
    InterviewCategory.TECHNICAL: 0.30,
    InterviewCategory.JOB_SPECIFIC: 0.20,
    InterviewCategory.RESUME: 0.10,
    InterviewCategory.PROJECT: 0.10,
    InterviewCategory.BEHAVIORAL: 0.20,
    InterviewCategory.SKILL_GAP: 0.10,
}


def plan_distribution(
    n: int,
    *,
    ratios: Optional[dict[InterviewCategory, float]] = None,
) -> list[InterviewCategory]:
    """Return a deterministic category distribution of length ``n``."""
    if n <= 0:
        return []
    r = ratios or DEFAULT_RATIOS
    counts = {c: max(0, round(n * w)) for c, w in r.items()}
    # Adjust to exactly n (rounding may under/over-shoot).
    total = sum(counts.values())
    order = list(r.keys())
    i = 0
    while total < n:
        counts[order[i % len(order)]] += 1
        total += 1
        i += 1
    while total > n:
        c = order[i % len(order)]
        if counts[c] > 0:
            counts[c] -= 1
            total -= 1
        i += 1
    out: list[InterviewCategory] = []
    for c in order:
        out.extend([c] * counts[c])
    return out[:n]


def build_questions(
    distribution: list[InterviewCategory],
    resume,
    job,
    match,
    skill_gap_report,
    *,
    difficulty: Difficulty = Difficulty.MEDIUM,
) -> list[InterviewQuestion]:
    """Fill the distribution with grounded fallback questions.

    Reassigns a slot if its category has no grounding data.
    """
    available = _available_categories(resume, job, match, skill_gap_report)
    questions: list[InterviewQuestion] = []
    used_skills: set[str] = set()

    for idx, cat in enumerate(distribution):
        # If the category has no data, reassign to an available one.
        if cat not in available:
            cat = _reassign(cat, available)
        q = _build_one(cat, idx, resume, job, match, skill_gap_report, difficulty, used_skills)
        if q is not None:
            questions.append(q)
    return questions


# --------------------------------------------------------------------------- #
# Availability + reassignment
# --------------------------------------------------------------------------- #


def _available_categories(resume, job, match, skill_gap_report) -> set[InterviewCategory]:
    avail: set[InterviewCategory] = set()
    if resume is not None:
        if getattr(resume, "skills", None):
            avail.add(InterviewCategory.TECHNICAL)
        if getattr(resume, "experience", None):
            avail.add(InterviewCategory.RESUME)
        if getattr(resume, "projects", None):
            avail.add(InterviewCategory.PROJECT)
        avail.add(InterviewCategory.BEHAVIORAL)  # always available
    if job is not None and getattr(job, "required_skills", None):
        avail.add(InterviewCategory.JOB_SPECIFIC)
    if (
        skill_gap_report is not None
        and getattr(skill_gap_report, "skill_gaps", None)
    ):
        avail.add(InterviewCategory.SKILL_GAP)
    return avail


def _reassign(cat: InterviewCategory, available: set[InterviewCategory]) -> InterviewCategory:
    if not available:
        return InterviewCategory.BEHAVIORAL
    # Prefer a "similar" fallback: technical -> resume -> behavioral.
    fallback_order = [
        InterviewCategory.TECHNICAL,
        InterviewCategory.RESUME,
        InterviewCategory.PROJECT,
        InterviewCategory.JOB_SPECIFIC,
        InterviewCategory.SKILL_GAP,
        InterviewCategory.BEHAVIORAL,
    ]
    for c in fallback_order:
        if c in available:
            return c
    return next(iter(available))


# --------------------------------------------------------------------------- #
# Question builders (grounded templates — no invented facts)
# --------------------------------------------------------------------------- #


def _build_one(cat, idx, resume, job, match, skill_gap_report, difficulty, used_skills):
    qid = f"IQ-{idx + 1:03d}"
    if cat is InterviewCategory.TECHNICAL:
        return _technical_question(qid, idx, resume, job, match, difficulty, used_skills)
    if cat is InterviewCategory.JOB_SPECIFIC:
        return _job_specific_question(qid, idx, job, difficulty, used_skills)
    if cat is InterviewCategory.RESUME:
        return _resume_question(qid, idx, resume, difficulty, used_skills)
    if cat is InterviewCategory.PROJECT:
        return _project_question(qid, idx, resume, difficulty, used_skills)
    if cat is InterviewCategory.SKILL_GAP:
        return _skill_gap_question(qid, idx, skill_gap_report, difficulty, used_skills)
    return _behavioral_question(qid, idx, difficulty)


def _technical_question(qid, idx, resume, job, match, difficulty, used_skills):
    # Prefer a matching skill (resume + job), else a resume skill.
    matching = list(getattr(match, "matching_skills", []) or []) if match else []
    resume_skills = list(getattr(resume, "skills", []) or []) if resume else []
    skill = _pick_skill(matching + resume_skills, used_skills)
    if skill is None:
        return _behavioral_question(qid, idx, difficulty)
    used_skills.add(skill.lower())
    source = "JOB" if skill in matching else "RESUME"
    return InterviewQuestion(
        question_id=qid,
        question=f"What role did {skill} play in your work, and why did you choose it?",
        category=InterviewCategory.TECHNICAL,
        difficulty=difficulty,
        target_skill=skill,
        source=source,
        rationale=f"Tests depth of understanding of {skill}, a skill present in the resume.",
        expected_evidence=f"Experience or projects mentioning {skill}.",
        related_job_requirement=skill if job and skill in (job.required_skills or []) else "",
    )


def _job_specific_question(qid, idx, job, difficulty, used_skills):
    required = list(getattr(job, "required_skills", []) or []) if job else []
    skill = _pick_skill(required, used_skills)
    if skill is None:
        return _behavioral_question(qid, idx, difficulty)
    used_skills.add(skill.lower())
    return InterviewQuestion(
        question_id=qid,
        question=(
            f"The role requires {skill}. How would you approach using "
            f"{skill} in this position, and what is your current level of "
            f"familiarity with it?"
        ),
        category=InterviewCategory.JOB_SPECIFIC,
        difficulty=difficulty,
        target_skill=skill,
        source="JOB",
        rationale=f"Assesses readiness for a required job skill ({skill}).",
        expected_evidence="Awareness of the skill; honest self-assessment is fine.",
        related_job_requirement=skill,
    )


def _resume_question(qid, idx, resume, difficulty, used_skills):
    experience = list(getattr(resume, "experience", []) or []) if resume else []
    if not experience:
        return _behavioral_question(qid, idx, difficulty)
    exp = experience[idx % len(experience)]
    company = getattr(exp, "company", "") or "your previous role"
    return InterviewQuestion(
        question_id=qid,
        question=f"Walk me through your role at {company} and your key contributions.",
        category=InterviewCategory.RESUME,
        difficulty=difficulty,
        target_skill="",
        source="RESUME",
        rationale="Probes depth of real experience listed on the resume.",
        expected_evidence=f"Experience at {company}.",
        related_job_requirement="",
    )


def _project_question(qid, idx, resume, difficulty, used_skills):
    projects = list(getattr(resume, "projects", []) or []) if resume else []
    if not projects:
        return _behavioral_question(qid, idx, difficulty)
    proj = projects[idx % len(projects)]
    name = getattr(proj, "name", "") or "your project"
    return InterviewQuestion(
        question_id=qid,
        question=f"Explain the architecture and your specific contribution to {name}.",
        category=InterviewCategory.PROJECT,
        difficulty=difficulty,
        target_skill=getattr(proj, "technologies", "") or "",
        source="PROJECT",
        rationale="Tests depth of understanding of a real resume project.",
        expected_evidence=f"Project: {name}.",
        related_job_requirement="",
    )


def _skill_gap_question(qid, idx, skill_gap_report, difficulty, used_skills):
    gaps = list(getattr(skill_gap_report, "skill_gaps", []) or []) if skill_gap_report else []
    if not gaps:
        return _behavioral_question(qid, idx, difficulty)
    gap = gaps[idx % len(gaps)]
    skill = getattr(gap, "skill", "") or "the skill"
    if skill.lower() in used_skills:
        # Find an unused gap skill.
        for g in gaps:
            if getattr(g, "skill", "").lower() not in used_skills:
                skill = g.skill
                break
    used_skills.add(skill.lower())
    return InterviewQuestion(
        question_id=qid,
        question=(
            f"What is your current understanding of {skill}, and what steps "
            f"are you taking to build proficiency in it?"
        ),
        category=InterviewCategory.SKILL_GAP,
        difficulty=difficulty,
        target_skill=skill,
        source="SKILL_GAP",
        rationale=f"Honestly assesses a known gap ({skill}); never claims the candidate has it.",
        expected_evidence="Self-assessment; learning plan is acceptable.",
        related_job_requirement=skill,
    )


def _behavioral_question(qid, idx, difficulty):
    templates = [
        "Tell me about a time you faced a significant technical challenge and how you resolved it.",
        "Describe a situation where you had to learn a new technology quickly. How did you approach it?",
        "Tell me about a time you disagreed with a teammate on a technical decision. How did you handle it?",
        "Describe a project that did not go as planned. What did you learn?",
    ]
    return InterviewQuestion(
        question_id=qid,
        question=templates[idx % len(templates)],
        category=InterviewCategory.BEHAVIORAL,
        difficulty=difficulty,
        target_skill="",
        source="GENERAL",
        rationale="Assesses reasoning, communication, and problem-solving.",
        expected_evidence="A real situation from the candidate's experience.",
        related_job_requirement="",
    )


def _pick_skill(skills, used_skills):
    for s in skills:
        if s and s.lower() not in used_skills:
            return s
    return None
