# CareerOS AI 🚀

**AI Career Intelligence & Interview Copilot**

An end-to-end AI career platform built with **Python** and **Streamlit** —
from resume intelligence to voice-based mock interviews.

> ⚠️ This repository currently contains the **clean project skeleton**
> only. No business logic and no AI API calls are wired up yet. The
> scaffold is intentionally modular so each planned feature can be
> implemented independently.

---

## ✨ Planned Modules

| # | Module | Description |
|---|-------|-------------|
| 1 | Resume Intelligence | Parse, analyze, and optimize resumes. |
| 2 | Job Description Analysis | Extract requirements & signals from JDs. |
| 3 | Skill Gap Analysis | Compare candidate skills vs. target roles. |
| 4 | Personalized Career Roadmap | Tailored milestone-based growth plans. |
| 5 | Application Copilot | Draft cover letters & outreach. |
| 6 | Retrieval-Augmented Generation (RAG) | Ground answers in user docs. |
| 7 | Voice AI Interview | Realistic voice mock interviews. |
| 8 | Salary Negotiation | Compensation benchmarking & scripts. |
| 9 | AI Evaluation Dashboard | Progress, quality & engagement metrics. |

---

## 🗂 Project Structure

```
careeros-ai/
├── app.py                       # Streamlit entry point
├── requirements.txt             # runtime deps
├── requirements-dev.txt         # dev deps (pytest)
├── .gitignore
├── README.md
├── .streamlit/
│   ├── config.toml              # Streamlit theme / server config
│   └── secrets.toml.example    # copy to secrets.toml & fill in
├── src/
│   ├── config/                  # app metadata & module registry
│   ├── core/                    # session state & shared utils
│   ├── ui/                      # layout, components, pages
│   ├── services/                # cross-cutting services (LLM, etc.)
│   └── modules/                 # one package per planned feature
│       ├── resume_intelligence/
│       ├── jd_analysis/
│       ├── skill_gap/
│       ├── career_roadmap/
│       ├── application_copilot/
│       ├── rag/
│       ├── voice_interview/
│       ├── salary_negotiation/
│       └── evaluation_dashboard/
├── docs/                        # architecture & roadmap notes
└── tests/                       # unit tests
```

---

## 🚀 Quick Start

```bash
# 1. Create & activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements-dev.txt

# 3. (optional) Configure secrets
cp .streamlit/secrets.toml.example .streamlit/secrets.toml

# 4. Run the app
streamlit run app.py
```

The app boots in **skeleton mode**: the sidebar lists every planned
module, and each module renders a placeholder card describing its scope.

---

## 🧪 Tests

```bash
pytest -q
```

---

## 📄 License

Proprietary — All rights reserved.
