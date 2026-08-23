"""Deterministic, conservative Evidence Validator.

Inspects generated application text (cover letters, tailored suggestions,
optimized bullets) against the candidate's evidence and returns a
:class:`EvidenceValidation` (PASS / WARNING / FAIL).

Conservative behavior (documented)
---------------------------------
The validator is intentionally conservative — it favors WARNING/FAIL when
it cannot confidently ground a claim. It does NOT perform perfect NLP;
instead it detects specific, high-risk claim categories:

+---------------------+-----------+--------------------------------------------+
| Claim category      | Verdict   | Detection                                  |
+---------------------+-----------+--------------------------------------------+
| Metric (40%, 5x,    | unsupported -> FAIL | the exact metric string is not in the |
| 100k, 1.2M)         |           | evidence corpus.                           |
| Year (2019, 2021)   | unsupported -> FAIL | the year is not in the evidence corpus. |
| Missing skill claimed| unsupported -> FAIL | a job-required skill marked missing in |
|                     |           | MatchAnalysis is claimed in the text.      |
| Certification claim | unsupported -> FAIL | "certified"/"certification" appears but no |
|                     |           | CERT evidence exists (or the cert name     |
|                     |           | is not in evidence).                       |
| Proper-noun entity  | warning -> WARNING  | a Capitalized word (>=3 chars, not a |
| (employer/title/proj)|           | stopword, not job-supplied) is not in the |
|                     |           | evidence corpus.                           |
+---------------------+-----------+--------------------------------------------+

Status aggregation:
- any unsupported_claims -> FAIL
- else any warnings -> WARNING
- else PASS

So unvalidated / fabricated content is NEVER silently approved.
"""

from __future__ import annotations

import re
from typing import Optional

from src.modules.application_copilot.evidence import evidence_corpus
from src.modules.application_copilot.model import (
    EvidenceItem,
    EvidenceValidation,
    ValidationStatus,
)
from src.modules.jd_analysis.model import JobAnalysis
from src.modules.resume_intelligence.model import ResumeAnalysis

# Common English stopwords excluded from the proper-noun check.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "the", "a", "an", "and", "or", "but", "with", "for", "to", "in", "of",
        "on", "at", "by", "as", "is", "are", "was", "were", "be", "been",
        "i", "my", "we", "our", "this", "that", "these", "those", "it",
        "its", "from", "have", "has", "had", "will", "would", "can", "could",
        "am", "me", "us", "you", "your", "he", "she", "they", "them",
        "not", "no", "so", "if", "then", "than", "also", "about", "into",
        "over", "under", "more", "most", "all", "any", "each", "which",
        "what", "who", "whom", "how", "why", "when", "where", "there",
        "here", "very", "too", "just", "now", "new", "use", "used", "using",
        "role", "position", "team", "work", "working", "experience",
        "skills", "skill", "ability", "strong", "background", "career",
        "opportunity", "interest", "excited", "passion", "passionate",
        "looking", "forward", "please", "thank", "thanks", "sincerely",
        "regards", "resume", "application", "apply", "candidate",
    }
)


def validate_claims(
    text: str,
    evidence: list[EvidenceItem],
    *,
    resume: Optional[ResumeAnalysis] = None,
    job: Optional[JobAnalysis] = None,
    match=None,
) -> EvidenceValidation:
    """Conservatively validate ``text`` against ``evidence``."""
    if not text or not text.strip():
        return EvidenceValidation(
            status=ValidationStatus.PASS,
            supported_claims=[],
            unsupported_claims=[],
            warnings=[],
        )

    corpus = evidence_corpus(evidence)
    supported: list[str] = []
    unsupported: list[str] = []
    warnings: list[str] = []

    # --- Metrics ---
    for m in _extract_metrics(text):
        if m.lower() in corpus:
            supported.append(f"Metric '{m}' supported by resume evidence.")
        else:
            unsupported.append(f"Metric '{m}' not found in resume evidence.")

    # --- Years ---
    for y in _extract_years(text):
        if y in corpus:
            supported.append(f"Year '{y}' supported by resume evidence.")
        else:
            unsupported.append(f"Year '{y}' not found in resume evidence.")

    # --- Missing skills claimed as possessed ---
    if match is not None and job is not None:
        missing = getattr(match, "missing_skills", []) or []
        text_tokens = _tokenize(text.lower())
        for s in job.required_skills:
            if _normalize(s) in {_normalize(m) for m in missing}:
                if _normalize(s) in text_tokens:
                    unsupported.append(
                        f"Technology '{s}' claimed but marked missing in resume evidence."
                    )

    # --- Certification claims ---
    text_lower = text.lower()
    if "certified" in text_lower or "certification" in text_lower:
        cert_evidence = [e for e in evidence if e.source_type == "certification"]
        if not cert_evidence:
            unsupported.append(
                "Certification claim made but no certification in resume evidence."
            )
        else:
            # Check that the SPECIFIC cert mentioned matches a resume cert.
            # Significant tokens of each cert (excluding generic words like
            # "certified"/"certification") must overlap the text; otherwise the
            # claimed cert is not one the candidate holds.
            text_tokens = _tokenize(text_lower)
            _cert_stop = {"certified", "certification", "the", "and", "a", "an", "of"}
            matched = False
            for ce in cert_evidence:
                cert_tokens = {
                    t
                    for t in _tokenize(ce.source_text.lower())
                    if len(t) >= 3 and t not in _cert_stop
                }
                if cert_tokens and cert_tokens & text_tokens:
                    matched = True
                    break
            if matched:
                supported.append("Certification claim supported by resume evidence.")
            else:
                unsupported.append(
                    "Certification claim not matched by any resume certification."
                )

    # --- Proper-noun entities (employers / titles / projects) ---
    stopwords = _stopword_set(job)
    for pn in _extract_proper_nouns(text):
        if pn.lower() in stopwords:
            continue
        if pn.lower() in corpus:
            supported.append(f"Entity '{pn}' supported by resume evidence.")
        else:
            warnings.append(
                f"Possible unsupported entity '{pn}' — not found in resume evidence."
            )

    # --- Aggregate status ---
    if unsupported:
        status = ValidationStatus.FAIL
    elif warnings:
        status = ValidationStatus.WARNING
    else:
        status = ValidationStatus.PASS

    return EvidenceValidation(
        status=status,
        supported_claims=supported,
        unsupported_claims=unsupported,
        warnings=warnings,
    )


# --------------------------------------------------------------------------- #
# Claim extractors
# --------------------------------------------------------------------------- #


def _extract_metrics(text: str) -> list[str]:
    """Extract percentage, multiplier, large-number, and count-claim strings.

    Count claims (e.g. "15 engineers", "10 people") catch invented
    team-size / responsibility metrics.
    """
    metrics: list[str] = []
    # Percentages: 40%, 99.9%
    for m in re.finditer(r"\d+(?:\.\d+)?\s*%", text):
        metrics.append(m.group(0).replace(" ", ""))
    # Multipliers: 5x, 10X (not part of a word)
    for m in re.finditer(r"\b\d+\s*[xX]\b", text):
        metrics.append(m.group(0).replace(" ", "").lower())
    # Large numbers: 100k, 1.2M, 5K
    for m in re.finditer(r"\b\d+(?:\.\d+)?\s*[kKmM]\b", text):
        metrics.append(m.group(0).replace(" ", "").lower())
    # Count claims: "15 engineers", "10 people", "5 team members"
    for m in re.finditer(
        r"\b\d+\s+(?:engineers?|people|members?|reports?|developers?|directs?)\b",
        text,
        re.IGNORECASE,
    ):
        metrics.append(m.group(0).lower())
    # Dedupe preserving order.
    seen: set[str] = set()
    out: list[str] = []
    for m in metrics:
        if m not in seen:
            seen.add(m)
            out.append(m)
    return out


def _extract_years(text: str) -> list[str]:
    """Extract 4-digit years (19xx / 20xx)."""
    seen: set[str] = set()
    out: list[str] = []
    for m in re.finditer(r"\b(?:19|20)\d{2}\b", text):
        if m.group(0) not in seen:
            seen.add(m.group(0))
            out.append(m.group(0))
    return out


def _extract_proper_nouns(text: str) -> list[str]:
    """Extract Capitalized words (>=3 chars) that may be entities.

    Excludes sentence-initial words conservatively (a capitalized word at
    the start of a sentence is likely a normal sentence start, not an
    entity). Acronyms (all-caps, e.g. AWS) are NOT flagged here — they are
    handled by the technology check.
    """
    out: list[str] = []
    # Split into sentences crudely on . ! ?
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for sent in sentences:
        words = re.findall(r"[A-Za-z][A-Za-z'\-]+", sent)
        if not words:
            continue
        for idx, w in enumerate(words):
            if len(w) < 3:
                continue
            if not w[0].isupper():
                continue
            if w.isupper():
                continue  # acronym — handled by tech check
            if idx == 0:
                continue  # sentence-initial — likely normal sentence start
            # Only flag if it has a lowercase variant (proper noun).
            out.append(w)
    # Dedupe preserving order.
    seen: set[str] = set()
    deduped: list[str] = []
    for w in out:
        key = w.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(w)
    return deduped


def _stopword_set(job: Optional[JobAnalysis]) -> set[str]:
    """Stopwords + job-supplied tokens (target role/company are fine to mention)."""
    words = set(_STOPWORDS)
    if job is not None:
        for field in (job.job_title, job.company, job.domain):
            if field:
                for tok in re.split(r"[^A-Za-z0-9+#]+", field):
                    if tok:
                        words.add(tok.lower())
    return words


def _normalize(s: str) -> str:
    if not s:
        return ""
    t = s.lower().strip()
    t = re.sub(r"[^a-z0-9+#]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _tokenize(text: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9+#]+", text) if t}
