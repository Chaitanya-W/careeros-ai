# Architecture

> Status: **Skeleton** — describes the intended design, not yet implemented.

## Overview

CareerOS AI is a Streamlit application organized as a modular Python
package. The entry point (`app.py`) stays thin: it configures the page,
initializes session state, and delegates rendering to UI helpers and
feature modules.

```
            ┌────────────────────────┐
            │         app.py          │  ← Streamlit entry point
            └───────────┬────────────┘
                        │
        ┌───────────────┼───────────────┐
        ▼               ▼               ▼
  src.config        src.core         src.ui
  (metadata,        (session        (layout,
   registry)        state, utils)    sidebar)
                        │
                        ▼
                ┌───────────────┐         ┌──────────────────┐
                │  src.modules  │ ◄──────►│  src.services    │
                │  (9 features)  │         │  (LLM, RAG, ...)  │
                └───────────────┘         └──────────────────┘
```

## Layering rules

1. **`app.py`** may import from `src.ui`, `src.core`, `src.config`.
2. **`src.ui`** may import from `src.config` and `src.core` — never
   from feature modules directly (navigation is data-driven).
3. **`src.modules.*`** may import from `src.services` and `src.core`.
   They should not import from one another to avoid coupling.
4. **`src.services`** holds cross-cutting service *interfaces*
   (LLM, vector store, etc.). Concrete providers are plugged in here.

## Module registry

Modules are declared once in `src/config/settings.py::MODULES`. Adding
a module is a three-step process:

1. Add an entry to `MODULES`.
2. Create a package under `src/modules/<key>/` exposing `render()`.
3. Wire the `render()` call from `app.py` (or a dispatcher in `src.ui`).

## Configuration & secrets

- Non-secret Streamlit config lives in `.streamlit/config.toml`.
- Secrets live in `.streamlit/secrets.toml` (git-ignored) using the
  template in `.streamlit/secrets.toml.example`.
