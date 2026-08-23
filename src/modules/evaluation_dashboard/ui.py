"""AI Evaluation Dashboard UI workflow.

Displays deterministic metrics, trends, answer quality, strengths,
improvement areas, and optional AI coaching insights. Includes JSON/CSV
export controls.

Empty states:
- No resume → "Analyze your resume first" + navigation.
- Resume but no interview → show available metrics; explain interview
  completion unlocks answer-quality metrics.
- Only one interview → show metrics; explain another session is needed
  for meaningful trends.

Partial data NEVER crashes the dashboard.
"""

from __future__ import annotations

import logging
from typing import Optional

import streamlit as st

from src.modules.evaluation_dashboard.exporter import export_csv, export_json
from src.modules.evaluation_dashboard.insights import enrich_insights, generate_insights
from src.modules.evaluation_dashboard.metrics import compute_metric_list, compute_metrics
from src.modules.evaluation_dashboard.model import EvaluationReport, EvaluationSummary, TrendPoint
from src.modules.evaluation_dashboard.trends import compute_trends
from src.modules.resume_intelligence.model import ResumeAnalysis
from src.services.gemini import GeminiConfigError, GeminiError
from src.ui.components.cards import empty_state_panel, section_header
from src.ui.components.cta import navigate_to

_logger = logging.getLogger(__name__)

_UNEXPECTED_ERROR_MESSAGE: str = (
    "Something went wrong while computing evaluation metrics. Please try again."
)


def render() -> None:
    """Render the Evaluation Dashboard."""
    section_header(
        "AI Evaluation Dashboard",
        "Track your progress, understand answer quality, and identify "
        "what to improve next.",
    )

    resume = st.session_state.get("resume_analysis")
    if not isinstance(resume, ResumeAnalysis):
        _render_no_resume_state()
        return

    _render_dashboard()


# --------------------------------------------------------------------------- #
# Empty states
# --------------------------------------------------------------------------- #


def _render_no_resume_state() -> None:
    empty_state_panel(
        title="Analyze your resume first.",
        hint="The Evaluation Dashboard tracks progress across your CareerOS "
        "modules. Head to Resume Intelligence to upload and analyze your resume.",
    )
    if st.button("Go to Resume Intelligence", type="primary", key="eval_goto_resume"):
        navigate_to("resume_intelligence")


# --------------------------------------------------------------------------- #
# Dashboard
# --------------------------------------------------------------------------- #


def _render_dashboard() -> None:
    # Compute metrics from current session state.
    if st.button("Refresh Metrics", type="primary", key="eval_refresh"):
        _compute_and_store()
        st.rerun()

    report = st.session_state.get("evaluation_report")
    if report is None:
        _compute_and_store()
        report = st.session_state.get("evaluation_report")

    if report is None or not isinstance(report, EvaluationReport):
        st.error("Could not compute evaluation metrics.")
        return

    summary = report.summary

    # --- Top metric cards ---
    _render_metric_cards(summary)

    # --- Progress trends ---
    _render_trends(report.trends)

    # --- Answer quality breakdown ---
    _render_answer_quality(summary)

    # --- Strengths + improvements ---
    _render_strengths_improvements(report)

    # --- Insights ---
    _render_insights(report)

    # --- Export controls ---
    _render_exports(report)


def _compute_and_store() -> None:
    try:
        session_dict = {k: st.session_state.get(k) for k in [
            "interview_report", "interview_evaluations", "interview_questions",
            "interview_session", "match_analysis", "skill_gap_report",
            "career_roadmap", "salary_benchmark", "salary_script",
            "salary_counter_offer", "resume_analysis",
        ]}
        summary = compute_metrics(session_dict)

        # Manage history for trends.
        metrics_state = st.session_state.get("evaluation_metrics")
        if not isinstance(metrics_state, dict):
            metrics_state = {"current": None, "history": []}
        history: list = metrics_state.get("history", [])
        # Only append if the new summary differs from the last.
        if not history or _summary_differs(summary, history[-1]):
            history.append(summary)
            # Keep last 20 snapshots.
            if len(history) > 20:
                history = history[-20:]
        metrics_state["current"] = summary
        metrics_state["history"] = history
        st.session_state["evaluation_metrics"] = metrics_state

        trends = compute_trends(history)
        st.session_state["evaluation_trends"] = trends

        insights = generate_insights(summary, trends)
        st.session_state["evaluation_insights"] = insights

        metrics_list = compute_metric_list(summary)
        strengths, improvements = _build_strengths_improvements(summary, trends)

        report = EvaluationReport(
            summary=summary,
            metrics=metrics_list,
            trends=trends,
            strengths=strengths,
            improvement_areas=improvements,
            insights=insights,
            generated_at=summary.generated_at,
            enriched=False,
        )
        st.session_state["evaluation_report"] = report
    except Exception:
        _logger.exception("Unexpected error computing evaluation metrics")
        st.error(_UNEXPECTED_ERROR_MESSAGE)


def _summary_differs(a: EvaluationSummary, b: EvaluationSummary) -> bool:
    """Check if two summaries differ in a meaningful way."""
    return (
        a.readiness_score != b.readiness_score
        or a.answer_quality.overall_avg != b.answer_quality.overall_avg
        or a.job_match_score != b.job_match_score
    )


# --------------------------------------------------------------------------- #
# Rendering
# --------------------------------------------------------------------------- #


def _render_metric_cards(summary: EvaluationSummary) -> None:
    section_header("Metrics", "Deterministic — not from a paid API or AI.")
    cols = st.columns(5)
    _metric_card(cols[0], "Readiness", summary.readiness_score, "%")
    _metric_card(cols[1], "Answer Quality", summary.answer_quality.overall_avg if summary.answer_quality.total_evaluated > 0 else None, "/10")
    _metric_card(cols[2], "Job Match", summary.job_match_score, "%")
    _metric_card(cols[3], "Skill Readiness", summary.skill_readiness, "%")
    _metric_card(cols[4], "Roadmap", summary.roadmap_readiness, "%")

    if summary.negotiation_prepared is not None:
        st.caption(f"Negotiation prepared: {'Yes' if summary.negotiation_prepared else 'No'}")

    if summary.interview_completed:
        st.caption(f"Interview: {summary.interview_completed_questions}/{summary.interview_total_questions} questions completed.")
    else:
        st.caption("No interview completed yet — answer-quality metrics will appear after your first interview.")


def _metric_card(col, label: str, value, unit: str) -> None:
    with col:
        with st.container(border=True):
            st.caption(label)
            if value is not None:
                st.markdown(f"## {value}{unit}")
            else:
                st.markdown("## —")
                st.caption("not available")


def _render_trends(trends: list[TrendPoint]) -> None:
    if not trends:
        return
    section_header("Progress Trends", "Across evaluation sessions (deterministic).")
    has_real_trends = any(t.direction.value != "insufficient_data" for t in trends)
    if not has_real_trends:
        st.info("Complete at least two interview sessions to unlock meaningful progress trends.")
        return
    for tp in trends:
        if tp.direction.value == "insufficient_data":
            continue
        with st.container(border=True):
            st.markdown(f"**{tp.label}**")
            st.caption(f"Direction: {tp.direction.value.upper()} | delta: {tp.delta}")
            if tp.values:
                st.caption(f"Values: {' → '.join(str(v) for v in tp.values)}")


def _render_answer_quality(summary: EvaluationSummary) -> None:
    aq = summary.answer_quality
    if aq.total_evaluated == 0:
        return
    section_header("Answer Quality Breakdown", "Heuristic coaching scores (0-10).")
    cols = st.columns(5)
    _aq_metric(cols[0], "Technical", aq.technical_avg)
    _aq_metric(cols[1], "Relevance", aq.relevance_avg)
    _aq_metric(cols[2], "Specificity", aq.specificity_avg)
    _aq_metric(cols[3], "Communication", aq.communication_avg)
    _aq_metric(cols[4], "Completeness", aq.completeness_avg)
    st.caption(f"Evidence alignment: {aq.evidence_alignment}")
    st.caption(f"Unsupported claims: {aq.unsupported_claim_count}")


def _aq_metric(col, label: str, value: float) -> None:
    with col:
        with st.container(border=True):
            st.metric(label, f"{value:.1f}")


def _render_strengths_improvements(report: EvaluationReport) -> None:
    col_left, col_right = st.columns(2)
    with col_left:
        if report.strengths:
            section_header("Strengths", None)
            for s in report.strengths:
                st.markdown(f"- {s}")
    with col_right:
        if report.improvement_areas:
            section_header("Improvement Areas", None)
            for s in report.improvement_areas:
                st.markdown(f"- {s}")


def _render_insights(report: EvaluationReport) -> None:
    if not report.insights:
        return
    section_header("AI Coaching Insights", "Deterministic + optional Gemini wording.")

    key_available = _gemini_key_available()
    if not key_available:
        st.info("AI insight refinement requires Gemini configuration. Deterministic insights are shown below.")
    elif not report.enriched:
        if st.button("Refine with AI", type="primary", key="eval_enrich"):
            _run_enrichment(report)
    else:
        st.caption("AI insight refinement: applied.")

    for insight in report.insights:
        st.markdown(f"- {insight}")


def _render_exports(report: EvaluationReport) -> None:
    section_header("Export", "Download deterministic evaluation data.")
    col_json, col_csv = st.columns(2)
    with col_json:
        if st.button("Export JSON", key="eval_export_json"):
            json_data = export_json(report)
            st.download_button(
                "Download JSON", data=json_data,
                file_name="careeros_evaluation.json", mime="application/json",
            )
    with col_csv:
        if st.button("Export CSV", key="eval_export_csv"):
            evaluations = st.session_state.get("interview_evaluations") or {}
            questions = st.session_state.get("interview_questions") or []
            csv_data = export_csv(evaluations, questions)
            st.download_button(
                "Download CSV", data=csv_data,
                file_name="careeros_answer_quality.csv", mime="text/csv",
            )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _build_strengths_improvements(summary: EvaluationSummary, trends: list[TrendPoint]) -> tuple[list[str], list[str]]:
    strengths: list[str] = []
    improvements: list[str] = []
    aq = summary.answer_quality

    if aq.total_evaluated > 0:
        scores = {
            "Technical": aq.technical_avg,
            "Relevance": aq.relevance_avg,
            "Specificity": aq.specificity_avg,
            "Communication": aq.communication_avg,
            "Completeness": aq.completeness_avg,
        }
        highest = max(scores, key=scores.get)
        lowest = min(scores, key=scores.get)
        strengths.append(f"{highest} is your strongest dimension ({scores[highest]:.1f}/10).")
        improvements.append(f"{lowest} needs practice ({scores[lowest]:.1f}/10).")

    for tp in trends:
        if tp.direction.value == "improving":
            strengths.append(f"{tp.label} is improving.")
        elif tp.direction.value == "declining":
            improvements.append(f"{tp.label} is declining — focus here.")

    if aq.unsupported_claim_count > 0:
        improvements.append(f"{aq.unsupported_claim_count} unsupported claim(s) — ground your answers in real experience.")

    if summary.negotiation_prepared is False:
        improvements.append("Negotiation preparation is incomplete.")

    if not strengths:
        strengths.append("Complete more CareerOS modules to build your strength profile.")
    if not improvements:
        improvements.append("No major improvement areas detected — keep practicing.")

    return strengths, improvements


def _run_enrichment(report: EvaluationReport) -> None:
    with st.spinner("Refining insights…"):
        try:
            from src.services.gemini import GeminiClient
            enriched = enrich_insights(
                report.insights, report.summary, report.trends,
                client=GeminiClient(),
            )
        except GeminiError as e:
            st.error(f"Gemini error: {e}")
            return
        except Exception:
            _logger.exception("Unexpected error during insight enrichment")
            st.error(_UNEXPECTED_ERROR_MESSAGE)
            return
    report.insights = enriched
    report.enriched = True
    st.session_state["evaluation_report"] = report
    st.success("AI insight refinement applied.")
    st.rerun()


def _gemini_key_available() -> bool:
    try:
        from src.services.gemini import get_api_key
        get_api_key()
        return True
    except GeminiConfigError:
        return False
    except Exception:
        _logger.exception("Unexpected error checking Gemini key")
        return False
