# agentic-ai

CLI-first **Agentic AI** portfolio repo (with minimal Streamlit UI) proving:
- agent workflows + tool calling
- RAG patterns (docs + retrieval + citations)
- evaluation harness (golden tests)
- guardrails (tool allowlists, step limits, redaction)
- reproducible DX (Makefile + CI)

## Quickstart (Windows)

```powershell
python -m venv .venv
.venv\Scripts\pip.exe install -U pip
.venv\Scripts\pip.exe install -e ".[dev]"
.venv\Scripts\python.exe -m pytest -q
.venv\Scripts\python.exe -m streamlit run apps\streamlit_app.py

# RAG v1 (Deterministic + Injection-Aware)

- Folder ingestion + chunking
- TF-IDF vectorization
- Cosine similarity ranking
- JSON index persistence
- Prompt-injection detection
- trusted_only retrieval mode