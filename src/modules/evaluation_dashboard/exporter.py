"""Evaluation report exporter — JSON + CSV.

Exports contain ONLY deterministic values. No secrets or API keys.
"""

from __future__ import annotations

import csv
import io
import json
from typing import Any

from src.modules.evaluation_dashboard.model import EvaluationReport


def export_json(report: EvaluationReport) -> str:
    """Export the full EvaluationReport as a JSON string."""
    return json.dumps(report.to_dict(), indent=2, ensure_ascii=False)


def export_csv(
    evaluations: dict,
    questions: list = None,
) -> str:
    """Export answer-level evaluation data as CSV.

    Each row: question_id, question, category, difficulty, answer,
    relevance_score, technical_score, specificity_score,
    communication_score, completeness_score, overall_score,
    evidence_alignment, unsupported_claims_count.
    """
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow([
        "question_id", "question", "category", "difficulty", "answer",
        "relevance_score", "technical_score", "specificity_score",
        "communication_score", "completeness_score", "overall_score",
        "evidence_alignment", "unsupported_claims_count",
    ])

    q_lookup: dict[str, Any] = {}
    if questions:
        for q in questions:
            qid = getattr(q, "question_id", "")
            if qid:
                q_lookup[qid] = q

    if not evaluations:
        return output.getvalue()

    evals = evaluations.values() if isinstance(evaluations, dict) else evaluations
    for ev in evals:
        qid = getattr(ev, "question_id", "")
        q = q_lookup.get(qid)
        writer.writerow([
            qid,
            getattr(q, "question", "") if q else "",
            getattr(getattr(q, "category", None), "value", "") if q and hasattr(q, "category") and hasattr(q.category, "value") else "",
            getattr(getattr(q, "difficulty", None), "value", "") if q and hasattr(q, "difficulty") and hasattr(q.difficulty, "value") else "",
            "",  # answer text not stored in evaluation; would need interview_answers
            getattr(ev, "relevance_score", 0.0),
            getattr(ev, "technical_score", 0.0),
            getattr(ev, "specificity_score", 0.0),
            getattr(ev, "communication_score", 0.0),
            getattr(ev, "completeness_score", 0.0),
            getattr(ev, "overall_score", 0.0),
            getattr(ev, "evidence_alignment", ""),
            len(getattr(ev, "unsupported_claims", []) or []),
        ])

    return output.getvalue()
