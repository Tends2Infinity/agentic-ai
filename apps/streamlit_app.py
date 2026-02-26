import streamlit as st

st.set_page_config(page_title="agentic-ai", layout="wide")

st.title("agentic-ai")
st.caption("CLI-first agentic AI portfolio with minimal Streamlit UI.")

st.markdown(
    """
### What you'll find here (built over 10 days)
- **Ops Copilot**: SOP runner, ticket triage, meeting-to-actions
- **Analytics Agent**: SQL reasoning + verification + decision narratives
- Shared **agent runtime**, **tooling**, **RAG**, and **evaluation harness**

Today (Day 1): repo scaffolding, packaging, DX, CI.
"""
)

st.divider()
st.subheader("Quickstart")
st.code(
    """python -m venv .venv
.venv\\Scripts\\pip.exe install -U pip
.venv\\Scripts\\pip.exe install -e ".[dev]"
.venv\\Scripts\\python.exe -m streamlit run apps\\streamlit_app.py""",
    language="bash",
)

st.info("Next: Day 2 adds the core agent runtime + tool registry.")