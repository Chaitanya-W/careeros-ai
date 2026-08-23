# Roadmap

> Status: **Skeleton mode** — no business logic implemented yet.

This document tracks the planned implementation order for the nine
CareerOS AI modules. Each milestone will be broken into focused tasks
when work begins.

## Milestone 0 — Foundation ✅

- [x] Repository skeleton & directory structure
- [x] Streamlit entry point (`app.py`)
- [x] Sidebar navigation + module registry
- [x] Session-state helper (`src.core.state`)
- [x] LLM service interface (`src.services.llm`) — placeholder

## Milestone 1 — Resume Intelligence

- [ ] Resume parsing (PDF / DOCX → structured data)
- [ ] ATS compatibility scoring
- [ ] Bullet-point rewrite suggestions

## Milestone 2 — Job Description Analysis

- [ ] JD parsing & section detection
- [ ] Skill & seniority signal extraction
- [ ] Red-flag detection

## Milestone 3 — Skill Gap Analysis

- [ ] Candidate ↔ role skill comparison
- [ ] Gap quantification & prioritization
- [ ] Learning recommendation engine

## Milestone 4 — Personalized Career Roadmap

- [ ] Milestone plan generation
- [ ] Curated learning resources
- [ ] Adaptive re-planning

## Milestone 5 — Application Copilot

- [ ] Cover-letter generation
- [ ] Outreach message drafting
- [ ] Application pipeline tracking

## Milestone 6 — Retrieval-Augmented Generation (RAG)

- [ ] Document ingestion & chunking
- [ ] Vector store integration
- [ ] Citation-grounded answers

## Milestone 7 — Voice AI Interview

- [ ] Speech-to-text capture
- [ ] Interview flow & question engine
- [ ] Real-time scoring & feedback

## Milestone 8 — Salary Negotiation

- [ ] Market compensation benchmarks
- [ ] Negotiation script generation
- [ ] Counter-offer rehearsal

## Milestone 9 — AI Evaluation Dashboard

- [ ] User progress metrics
- [ ] AI answer quality metrics
- [ ] Aggregate trends & exports
