"""Shared pytest fixtures for the Resume Intelligence tests.

Fixtures generate small in-memory resume files (TXT/DOCX/PDF) and
malformed variants so tests do not depend on committed binary files.
A canned Gemini JSON response + a fake client let analyzer/UI tests run
without a real API key.
"""

from __future__ import annotations

import io
import json
from typing import Optional

import pytest


# --------------------------------------------------------------------------- #
# TXT fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def sample_txt_bytes() -> bytes:
    return (
        b"Jane Smith\n"
        b"Senior Software Engineer\n\n"
        b"Skills: Python, Go, PostgreSQL, Kubernetes\n\n"
        b"Experience\n"
        b"Senior Engineer at Acme Corp, 2021-present: Built distributed APIs.\n"
        b"Engineer at Globex, 2018-2021: Maintained billing services.\n\n"
        b"Education\n"
        b"B.S. Computer Science, MIT, 2018\n"
    )


@pytest.fixture
def empty_txt_bytes() -> bytes:
    return b"   \n\t  "


# --------------------------------------------------------------------------- #
# DOCX fixtures (generated with python-docx)
# --------------------------------------------------------------------------- #


@pytest.fixture
def sample_docx_bytes() -> bytes:
    import docx

    buf = io.BytesIO()
    doc = docx.Document()
    doc.add_paragraph("Alex Lee")
    doc.add_paragraph("Backend Engineer")
    doc.add_paragraph("Skills: Go, Kubernetes, PostgreSQL")
    doc.add_paragraph(
        "Experience: 3 years at Initech building distributed systems."
    )
    doc.save(buf)
    return buf.getvalue()


@pytest.fixture
def malformed_docx_bytes() -> bytes:
    # Not a valid DOCX (which is a zip) — python-docx will reject it.
    return b"this is not a valid docx file"


# --------------------------------------------------------------------------- #
# PDF fixtures (generated with fpdf2)
# --------------------------------------------------------------------------- #


@pytest.fixture
def sample_pdf_bytes() -> bytes:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", size=12)
    # Use cell() with new_x="LMARGIN" so the cursor resets to the left
    # margin after each line (multi_cell(w=0) leaves x at the right margin
    # and the next call has no horizontal space).
    for line in (
        "Sam Taylor",
        "Data Scientist",
        "Skills: Python, Pandas, TensorFlow, SQL",
        "Experience: 4 years at Umbrella doing ML in production.",
    ):
        pdf.cell(0, 8, text=line, new_x="LMARGIN", new_y="NEXT")
    return bytes(pdf.output())


@pytest.fixture
def malformed_pdf_bytes() -> bytes:
    # Starts like a PDF header but is not a valid PDF structure.
    return b"%PDF-1.4 this is not really a valid pdf body structure"


@pytest.fixture
def empty_bytes() -> bytes:
    return b""


# --------------------------------------------------------------------------- #
# Gemini canned response + fake client (NO real API key needed)
# --------------------------------------------------------------------------- #


@pytest.fixture
def sample_analysis_dict() -> dict:
    """A complete, valid Gemini-style analysis payload."""
    return {
        "overall_score": 78,
        "professional_summary": (
            "Backend engineer with 5 years building distributed systems."
        ),
        "skills": ["Python", "Go", "PostgreSQL", "Kubernetes"],
        "experience": [
            {
                "title": "Senior Engineer",
                "company": "Acme Corp",
                "duration": "2021-present",
                "description": "Built and operated distributed APIs.",
            }
        ],
        "education": [
            {
                "degree": "B.S. Computer Science",
                "institution": "MIT",
                "year": "2018",
                "details": "",
            }
        ],
        "projects": [],
        "certifications": [],
        "strengths": ["Strong systems design", "Quantified impact"],
        "weaknesses": ["Sparse project detail"],
        "improvement_suggestions": [
            "Add more metrics to experience bullets.",
            "Include a projects section.",
        ],
        "recommended_skills": ["Terraform", "Rust"],
    }


@pytest.fixture
def sample_analysis_json(sample_analysis_dict: dict) -> str:
    return json.dumps(sample_analysis_dict)


class FakeGeminiClient:
    """Test double for :class:`GeminiClient` — returns a canned response.

    ``raise_exc`` (if set) is raised by ``generate_json`` to simulate an
    API / config / parse error without touching the network.
    """

    def __init__(
        self,
        response: str = "",
        *,
        raise_exc: Optional[Exception] = None,
    ) -> None:
        self._response = response
        self._raise = raise_exc
        self.calls: list[dict] = []

    def generate_json(
        self,
        system_prompt: str,
        user_text: str,
        response_schema: Optional[dict] = None,
        *,
        temperature: float = 0.2,
    ) -> str:
        self.calls.append(
            {
                "system_prompt": system_prompt,
                "user_text": user_text,
                "response_schema": response_schema,
                "temperature": temperature,
            }
        )
        if self._raise is not None:
            raise self._raise
        return self._response


@pytest.fixture
def fake_gemini_client_factory():
    """Return a factory that builds :class:`FakeGeminiClient` instances."""
    return FakeGeminiClient


@pytest.fixture
def app_path() -> str:
    """Absolute path to app.py for AppTest."""
    from pathlib import Path

    return str(Path(__file__).resolve().parent.parent / "app.py")


# --------------------------------------------------------------------------- #
# Job Match fixtures — canned JobAnalysis JSON, sample resume, fake client
# (reuses the FakeGeminiClient pattern from Step 3).
# --------------------------------------------------------------------------- #


@pytest.fixture
def sample_job_analysis_dict() -> dict:
    """A complete, valid Gemini-style JobAnalysis payload."""
    return {
        "job_title": "Senior Backend Engineer",
        "company": "Acme Corp",
        "required_skills": ["Python", "SQL", "AWS", "Docker"],
        "preferred_skills": ["Kubernetes", "Terraform"],
        "responsibilities": [
            "Design and operate distributed services.",
            "Own backend reliability and performance.",
        ],
        "experience_requirements": "5+ years of backend engineering",
        "education_requirements": "B.S. in Computer Science or equivalent",
        "keywords": ["Python", "AWS", "distributed systems", "reliability"],
        "domain": "Backend Engineering",
    }


@pytest.fixture
def sample_job_analysis_json(sample_job_analysis_dict: dict) -> str:
    return json.dumps(sample_job_analysis_dict)


@pytest.fixture
def sample_resume_for_matching():
    """A ResumeAnalysis used to exercise the deterministic matcher."""
    from src.modules.resume_intelligence.model import (
        EducationItem,
        ExperienceItem,
        ResumeAnalysis,
    )

    return ResumeAnalysis(
        overall_score=80,
        professional_summary="Backend engineer with 6 years building services.",
        skills=["Python", "SQL", "Pandas", "Git", "AWS Lambda", "React"],
        experience=[
            ExperienceItem(
                title="Senior Engineer",
                company="Acme Corp",
                duration="2019-present",
                description="Built distributed Python services on AWS.",
            )
        ],
        education=[
            EducationItem(
                degree="B.S. Computer Science",
                institution="MIT",
                year="2019",
                details="",
            )
        ],
    )


# --------------------------------------------------------------------------- #
# Skill Gap fixtures — predictable missing_skills + enrichment payloads.
# --------------------------------------------------------------------------- #


@pytest.fixture
def sample_skill_gap_resume():
    """A resume with [Python, SQL] so required gaps = [AWS, Docker]."""
    from src.modules.resume_intelligence.model import (
        EducationItem,
        ExperienceItem,
        ResumeAnalysis,
    )

    return ResumeAnalysis(
        overall_score=70,
        professional_summary="Backend engineer.",
        skills=["Python", "SQL"],
        experience=[
            ExperienceItem(
                title="Engineer",
                company="Acme",
                duration="2019-present",
                description="Built Python services.",
            )
        ],
        education=[
            EducationItem(
                degree="B.S. Computer Science",
                institution="MIT",
                year="2019",
                details="",
            )
        ],
    )


@pytest.fixture
def sample_skill_gap_inputs(sample_skill_gap_resume):
    """(resume, job, match) with missing_skills == ['AWS', 'Docker'] and
    preferred gaps == ['Kubernetes', 'Terraform']."""
    from src.modules.jd_analysis.matcher import match_jobs
    from src.modules.jd_analysis.model import JobAnalysis

    job = JobAnalysis(
        job_title="Senior Backend Engineer",
        company="Acme Corp",
        required_skills=["Python", "SQL", "AWS", "Docker"],
        preferred_skills=["Kubernetes", "Terraform"],
        responsibilities=["Build distributed services."],
        experience_requirements="5+ years",
        education_requirements="B.S. in Computer Science",
        keywords=["Python", "AWS", "distributed"],
        domain="Backend Engineering",
    )
    match = match_jobs(sample_skill_gap_resume, job)
    return sample_skill_gap_resume, job, match


@pytest.fixture
def sample_enrichment_json() -> str:
    """A valid Gemini enrichment JSON for the supplied skills only."""
    import json

    return json.dumps(
        {
            "skills": [
                {
                    "skill": "Docker",
                    "why_it_matters": "Containers are essential for portable deployments.",
                    "learning_objectives": [
                        "Understand images and containers",
                        "Write a Dockerfile",
                    ],
                    "learning_path": [
                        "Read the getting started guide",
                        "Containerize a small app",
                    ],
                    "practice_project": "Dockerize a Python web service.",
                    "estimated_effort": "1-2 weeks of part-time study.",
                },
                {
                    "skill": "Kubernetes",
                    "why_it_matters": "Orchestration scales containers in production.",
                    "learning_objectives": ["Understand pods and deployments"],
                    "learning_path": ["Run minikube", "Deploy a sample app"],
                    "practice_project": "Deploy a service to a local cluster.",
                    "estimated_effort": "2-4 weeks of part-time study.",
                },
            ]
        }
    )


@pytest.fixture
def skill_adding_enrichment_json() -> str:
    """An enrichment JSON that attempts to ADD a skill not in the supplied
    list (must be rejected by the enricher)."""
    import json

    return json.dumps(
        {
            "skills": [
                {
                    "skill": "Docker",
                    "why_it_matters": "Containers matter.",
                    "learning_objectives": [],
                    "learning_path": [],
                    "practice_project": "",
                    "estimated_effort": "",
                },
                {
                    "skill": "InventedSkill",
                    "why_it_matters": "AI invented this skill.",
                    "learning_objectives": ["fake"],
                    "learning_path": [],
                    "practice_project": "",
                    "estimated_effort": "",
                },
            ]
        }
    )


# --------------------------------------------------------------------------- #
# Career Roadmap fixtures — SkillGapReport + CareerRoadmap + enrichment
# payloads (the enrichment payloads are built dynamically from the actual
# roadmap to guarantee they match its skill structure).
# --------------------------------------------------------------------------- #


@pytest.fixture
def sample_skill_gap_report(sample_skill_gap_inputs):
    """A SkillGapReport built deterministically from the sample inputs."""
    from src.modules.skill_gap.report import build_skill_gap_report

    resume, job, match = sample_skill_gap_inputs
    return build_skill_gap_report(resume, job, match)


@pytest.fixture
def sample_career_roadmap(sample_skill_gap_report, sample_skill_gap_inputs):
    """A deterministic CareerRoadmap built from the SkillGapReport."""
    from src.modules.career_roadmap.builder import build_career_roadmap

    _resume, job, match = sample_skill_gap_inputs
    return build_career_roadmap(sample_skill_gap_report, match, job)


def _roadmap_enrichment_for(roadmap) -> dict:
    """Build a VALID enrichment dict matching the roadmap's skill structure."""
    return {
        "estimated_total_effort": "8-16 weeks of focused study.",
        "final_outcome": "Reach job-ready alignment with refined milestones.",
        "phases": [
            {
                "phase_number": p.phase_number,
                "skills": list(p.skills),  # exact same order
                "objective": f"Refined objective for {', '.join(p.skills)}.",
                "duration": f"{p.phase_number * 3} weeks (refined).",
                "practice_project": f"Refined project using {', '.join(p.skills)}.",
                "milestones": [
                    {
                        "skill": ms.skill,
                        "title": f"Refined: {ms.title}",
                        "description": f"Refined description for {ms.skill}.",
                        "completion_criteria": f"Refined criteria for {ms.skill}.",
                    }
                    for ms in p.milestones
                ],
            }
            for p in roadmap.phases
        ],
    }


@pytest.fixture
def sample_roadmap_enrichment_json(sample_career_roadmap) -> str:
    """A VALID Gemini enrichment JSON for the sample roadmap."""
    import json

    return json.dumps(_roadmap_enrichment_for(sample_career_roadmap))


@pytest.fixture
def roadmap_skill_adding_json(sample_career_roadmap) -> str:
    """Enrichment JSON that ADDS a skill to phase 1 (must be rejected)."""
    import json

    data = _roadmap_enrichment_for(sample_career_roadmap)
    data["phases"][0]["skills"] = list(data["phases"][0]["skills"]) + ["InventedSkill"]
    return json.dumps(data)


@pytest.fixture
def roadmap_skill_removing_json(sample_career_roadmap) -> str:
    """Enrichment JSON that REMOVES a skill from phase 1 (must be rejected)."""
    import json

    data = _roadmap_enrichment_for(sample_career_roadmap)
    if len(data["phases"][0]["skills"]) > 1:
        data["phases"][0]["skills"] = data["phases"][0]["skills"][:-1]
    return json.dumps(data)


@pytest.fixture
def roadmap_skill_reordering_json(sample_career_roadmap) -> str:
    """Enrichment JSON that REORDERS skills in phase 1 (must be rejected)."""
    import json

    data = _roadmap_enrichment_for(sample_career_roadmap)
    skills = data["phases"][0]["skills"]
    if len(skills) > 1:
        data["phases"][0]["skills"] = list(reversed(skills))
    return json.dumps(data)


# --------------------------------------------------------------------------- #
# Application Copilot fixtures — a richer resume (with project + cert) for
# evidence extraction + validation tests.
# --------------------------------------------------------------------------- #


@pytest.fixture
def sample_copilot_resume():
    """A ResumeAnalysis with experience, project, skills, education, cert."""
    from src.modules.resume_intelligence.model import (
        CertificationItem,
        EducationItem,
        ExperienceItem,
        ProjectItem,
        ResumeAnalysis,
    )

    return ResumeAnalysis(
        overall_score=72,
        professional_summary="Backend engineer building Python services on AWS.",
        skills=["Python", "SQL", "AWS"],
        experience=[
            ExperienceItem(
                title="Software Engineer",
                company="Acme",
                duration="2019-present",
                description="Built Python services on AWS serving 100k users.",
            )
        ],
        education=[
            EducationItem(
                degree="B.S. Computer Science",
                institution="MIT",
                year="2019",
                details="",
            )
        ],
        projects=[
            ProjectItem(
                name="CareerOS",
                description="Built CareerOS using Python and Streamlit.",
                technologies="Python, Streamlit",
            )
        ],
        certifications=[
            CertificationItem(name="AWS Certified Developer", issuer="Amazon", year="2022")
        ],
    )


@pytest.fixture
def sample_copilot_job():
    from src.modules.jd_analysis.model import JobAnalysis

    return JobAnalysis(
        job_title="Senior Backend Engineer",
        company="Globex",
        required_skills=["Python", "SQL", "AWS", "Docker"],
        preferred_skills=["Kubernetes"],
        responsibilities=["Build distributed services."],
        experience_requirements="5+ years",
        education_requirements="B.S. in Computer Science",
        keywords=["Python", "AWS", "distributed"],
        domain="Backend Engineering",
    )


@pytest.fixture
def sample_copilot_match(sample_copilot_resume, sample_copilot_job):
    from src.modules.jd_analysis.matcher import match_jobs

    return match_jobs(sample_copilot_resume, sample_copilot_job)


@pytest.fixture
def sample_copilot_inputs(
    sample_copilot_resume, sample_copilot_job, sample_copilot_match
):
    """(resume, job, match, evidence)."""
    from src.modules.application_copilot.evidence import extract_evidence

    evidence = extract_evidence(sample_copilot_resume)
    return sample_copilot_resume, sample_copilot_job, sample_copilot_match, evidence


# --------------------------------------------------------------------------- #
# RAG fixtures — markdown bytes + a small indexed corpus + canned answers.
# --------------------------------------------------------------------------- #


@pytest.fixture
def sample_md_bytes() -> bytes:
    return (
        b"# CareerOS Guide\n\n"
        b"The CareerOS project uses Python and Streamlit.\n\n"
        b"It helps users build career intelligence dashboards.\n"
    )


@pytest.fixture
def sample_md_bytes_alt() -> bytes:
    return b"# Notes\n\nDocker containers package applications for portable deployment.\n"


@pytest.fixture
def rag_embedder():
    from src.modules.rag.embedder import LocalEmbedder

    return LocalEmbedder()


@pytest.fixture
def rag_index_with_docs(rag_embedder):
    """A VectorIndex with two documents (A: CareerOS/Python/Streamlit,
    B: Docker), plus the documents/chunks dicts and an id map."""
    from src.modules.rag.chunker import chunk_text, make_chunk_id
    from src.modules.rag.model import DocumentChunk
    from src.modules.rag.store import VectorIndex

    idx = VectorIndex()
    docs = {}
    chunks_map = {}
    corpus = {
        "docA": ("careeros.txt", "The CareerOS project uses Python and Streamlit."),
        "docB": ("docker.txt", "Docker containers package applications for portable deployment."),
    }
    for doc_id, (filename, text) in corpus.items():
        parts = chunk_text(text)
        ch_objs = [
            DocumentChunk(
                chunk_id=make_chunk_id(doc_id, i),
                document_id=doc_id,
                chunk_index=i,
                text=c,
                metadata={"filename": filename, "page": 1, "chunk_index": i, "document_id": doc_id},
            )
            for i, c in enumerate(parts)
        ]
        for ch in ch_objs:
            idx.add(ch, rag_embedder.embed(ch.text))
        chunks_map[doc_id] = ch_objs
        docs[doc_id] = filename
    return idx, docs, chunks_map


@pytest.fixture
def rag_index_careeros(rag_embedder):
    """A single-document index with the exact spec text."""
    from src.modules.rag.chunker import chunk_text, make_chunk_id
    from src.modules.rag.model import DocumentChunk
    from src.modules.rag.store import VectorIndex

    idx = VectorIndex()
    text = "The CareerOS project uses Python and Streamlit."
    parts = chunk_text(text)
    doc_id = "docA"
    for i, c in enumerate(parts):
        ch = DocumentChunk(
            chunk_id=make_chunk_id(doc_id, i),
            document_id=doc_id,
            chunk_index=i,
            text=c,
            metadata={"filename": "careeros.txt", "page": 1, "chunk_index": i, "document_id": doc_id},
        )
        idx.add(ch, rag_embedder.embed(ch.text))
    return idx, doc_id, "careeros.txt"


@pytest.fixture
def rag_grounded_json() -> str:
    import json

    return json.dumps(
        {
            "answer": "The project uses Python and Streamlit. [1]",
            "cited_source_ids": ["1"],
            "insufficient_context": False,
        }
    )


@pytest.fixture
def rag_invalid_citation_json() -> str:
    import json

    return json.dumps(
        {
            "answer": "The project uses AWS. [FAKE-123]",
            "cited_source_ids": ["FAKE-123"],
            "insufficient_context": False,
        }
    )


# --------------------------------------------------------------------------- #
# AI Interview Coach fixtures.
# --------------------------------------------------------------------------- #


@pytest.fixture
def sample_interview_inputs(
    sample_copilot_resume, sample_copilot_job, sample_copilot_match
):
    """(resume, job, match, skill_gap_report, evidence)."""
    from src.modules.application_copilot.evidence import extract_evidence
    from src.modules.skill_gap.report import build_skill_gap_report

    sgr = build_skill_gap_report(sample_copilot_resume, sample_copilot_job, sample_copilot_match)
    evidence = extract_evidence(sample_copilot_resume)
    return sample_copilot_resume, sample_copilot_job, sample_copilot_match, sgr, evidence


@pytest.fixture
def interview_question_gen_json() -> str:
    """A VALID Gemini question-generation response (grounded)."""
    import json

    return json.dumps(
        {
            "questions": [
                {
                    "question": "What role did Python play in your work, and why did you choose it?",
                    "category": "technical",
                    "difficulty": "medium",
                    "target_skill": "Python",
                    "rationale": "Tests depth of Python understanding.",
                },
                {
                    "question": "Explain the architecture and your contribution to CareerOS.",
                    "category": "project",
                    "difficulty": "medium",
                    "target_skill": "Python",
                    "rationale": "Tests project depth.",
                },
            ]
        }
    )


@pytest.fixture
def interview_question_invented_json() -> str:
    """A Gemini response that INVENTS a skill (TensorFlow) — must be rejected."""
    import json

    return json.dumps(
        {
            "questions": [
                {
                    "question": "You used TensorFlow extensively in your projects. Explain how.",
                    "category": "technical",
                    "difficulty": "medium",
                    "target_skill": "TensorFlow",
                    "rationale": "fake",
                }
            ]
        }
    )


@pytest.fixture
def interview_eval_json() -> str:
    """A VALID Gemini answer-evaluation response."""
    import json

    return json.dumps(
        {
            "relevance_score": 8,
            "technical_score": 7,
            "specificity_score": 6,
            "communication_score": 7,
            "completeness_score": 6,
            "overall_score": 7.0,
            "strengths": ["Clear explanation.", "Addresses the question."],
            "improvements": ["Add a concrete metric."],
            "missing_points": ["Specific outcome."],
            "evidence_alignment": "aligned",
            "unsupported_claims": [],
            "recommendation": "Good answer; add measurable outcomes.",
        }
    )


# --------------------------------------------------------------------------- #
# Salary Negotiator fixtures.
# --------------------------------------------------------------------------- #


@pytest.fixture
def sample_salary_inputs(
    sample_copilot_resume, sample_copilot_job, sample_copilot_match
):
    """(resume, job, match, evidence) for salary negotiation tests."""
    from src.modules.application_copilot.evidence import extract_evidence

    evidence = extract_evidence(sample_copilot_resume)
    return sample_copilot_resume, sample_copilot_job, sample_copilot_match, evidence


@pytest.fixture
def salary_enrichment_valid_json() -> str:
    """A VALID Gemini enrichment response (improved wording only — no
    invented dollar numbers or candidate facts)."""
    import json

    return json.dumps(
        {
            "talking_points": [
                {"point_id": "NP-001", "wording": "I've researched the market for this role and I'm looking for fair compensation."},
                {"point_id": "NP-002", "wording": "I bring proven experience with Python, SQL, and AWS — core requirements for this position."},
                {"point_id": "NP-003", "wording": "In my role at Acme, I built Python services on AWS."},
                {"point_id": "NP-004", "wording": "Given the market range and my experience, I'd like to propose the counter number as a base salary."},
                {"point_id": "NP-005", "wording": "I'm also open to equity, signing bonuses, or performance-based incentives."},
                {"point_id": "NP-006", "wording": "I'm excited about the opportunity and confident I can deliver significant value."},
            ]
        }
    )


@pytest.fixture
def salary_enrichment_invented_json() -> str:
    """A Gemini enrichment that INVENTS a metric — must be rejected."""
    import json

    return json.dumps(
        {
            "talking_points": [
                {"point_id": "NP-001", "wording": "I increased revenue by 500% at my previous role."},
                {"point_id": "NP-002", "wording": "I bring proven experience with Python."},
                {"point_id": "NP-003", "wording": "In my role at Acme, I built Python services."},
                {"point_id": "NP-004", "wording": "I'd like to propose a base salary of $999,999."},
                {"point_id": "NP-005", "wording": "I'm open to equity."},
                {"point_id": "NP-006", "wording": "I'm excited."},
            ]
        }
    )


# --------------------------------------------------------------------------- #
# Evaluation Dashboard fixtures.
# --------------------------------------------------------------------------- #


@pytest.fixture
def sample_eval_session():
    """A dict mimicking st.session_state with interview/match/skill/roadmap/salary."""
    from types import SimpleNamespace
    from src.modules.voice_interview.model import (
        AnswerEvaluation,
        InterviewCategory,
        InterviewQuestion,
        InterviewReport,
    )

    evals = {
        "IQ-001": AnswerEvaluation(
            question_id="IQ-001", relevance_score=7.0, technical_score=6.0,
            specificity_score=5.0, communication_score=8.0, completeness_score=6.0,
            overall_score=6.5, evidence_alignment="aligned",
            unsupported_claims=[], recommendation="Good.",
        ),
        "IQ-002": AnswerEvaluation(
            question_id="IQ-002", relevance_score=8.0, technical_score=7.0,
            specificity_score=6.0, communication_score=7.0, completeness_score=7.0,
            overall_score=7.0, evidence_alignment="aligned",
            unsupported_claims=[], recommendation="OK.",
        ),
    }
    questions = [
        InterviewQuestion(question_id="IQ-001", category=InterviewCategory.TECHNICAL),
        InterviewQuestion(question_id="IQ-002", category=InterviewCategory.BEHAVIORAL),
    ]
    report = InterviewReport(
        readiness_score=72, technical_score=6.5, behavioral_score=7.0,
        communication_score=7.5, job_alignment_score=6.8,
        completed_questions=2, total_questions=2,
    )
    match = SimpleNamespace(
        overall_match_score=68, matching_skills=["Python", "SQL"],
        missing_skills=["Docker"],
    )
    skill_gap = SimpleNamespace(
        high_priority_count=1, medium_priority_count=1, low_priority_count=0,
    )
    roadmap = SimpleNamespace(current_readiness=68)
    salary_bm = SimpleNamespace(mid=140000, currency="USD", currency_symbol="$")
    salary_script = SimpleNamespace(enriched=False)
    salary_counter = SimpleNamespace(counter_number=140000)

    return {
        "interview_report": report,
        "interview_evaluations": evals,
        "interview_questions": questions,
        "match_analysis": match,
        "skill_gap_report": skill_gap,
        "career_roadmap": roadmap,
        "salary_benchmark": salary_bm,
        "salary_script": salary_script,
        "salary_counter_offer": salary_counter,
    }


@pytest.fixture
def sample_eval_history():
    """A 3-snapshot EvaluationSummary history for trend tests."""
    from src.modules.evaluation_dashboard.model import AnswerQuality, EvaluationSummary

    return [
        EvaluationSummary(
            readiness_score=55, answer_quality=AnswerQuality(
                technical_avg=5.0, communication_avg=6.0, specificity_avg=4.0,
                completeness_avg=5.0, overall_avg=5.5, total_evaluated=2,
            ),
        ),
        EvaluationSummary(
            readiness_score=65, answer_quality=AnswerQuality(
                technical_avg=6.0, communication_avg=7.0, specificity_avg=5.0,
                completeness_avg=6.0, overall_avg=6.5, total_evaluated=2,
            ),
        ),
        EvaluationSummary(
            readiness_score=72, answer_quality=AnswerQuality(
                technical_avg=6.5, communication_avg=7.5, specificity_avg=5.5,
                completeness_avg=6.5, overall_avg=7.0, total_evaluated=2,
            ),
        ),
    ]


@pytest.fixture
def eval_insight_valid_json() -> str:
    """A VALID Gemini insight enrichment (same numbers, improved wording)."""
    import json
    return json.dumps({
        "insights": [
            "Interview readiness improved from 55% to 72%.",
            "Specificity is currently your lowest-scoring answer dimension (5.5/10).",
            "Technical is your strongest dimension (6.5/10).",
        ]
    })


@pytest.fixture
def eval_insight_changed_numbers_json() -> str:
    """A Gemini insight that CHANGES numbers — must be rejected."""
    import json
    return json.dumps({
        "insights": [
            "Interview readiness improved from 55% to 999%.",
            "Specificity is your lowest dimension (1.0/10).",
            "Technical is your strongest (10.0/10).",
        ]
    })
