import streamlit as st
import requests
import json
import time

st.set_page_config(page_title="dotkonnekt", page_icon="📄", layout="wide")

st.title("dotkonnekt")
st.caption("Document ingestion, analysis & retrieval")

# --- Sidebar: API config ---
with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("API Base URL", value="http://localhost:8000/api/v1")

    st.divider()
    if st.button("Check Health"):
        try:
            r = requests.get(f"{api_url}/health", timeout=5)
            if r.status_code == 200:
                st.success("API is healthy")
            else:
                st.error(f"API returned {r.status_code}")
        except requests.ConnectionError:
            st.error("Cannot connect to API")

# --- Session state init ---
if "job_id" not in st.session_state:
    st.session_state.job_id = None
if "job_status" not in st.session_state:
    st.session_state.job_status = None
if "report" not in st.session_state:
    st.session_state.report = None

# --- Tabs ---
upload_tab, status_tab, report_tab, query_tab = st.tabs(["Upload", "Status", "Report", "Query"])

# === UPLOAD TAB ===
with upload_tab:
    st.subheader("Upload Documents")
    files = st.file_uploader(
        "Drag & drop PDF files here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    collection = st.text_input("Collection name (optional)", placeholder="e.g. legal-docs")

    if st.button("Ingest", type="primary", disabled=not files):
        with st.spinner("Uploading..."):
            upload_files = [("files", (f.name, f.getvalue(), "application/pdf")) for f in files]
            data = {}
            if collection:
                data["collection"] = collection

            try:
                r = requests.post(
                    f"{api_url}/documents",
                    files=upload_files,
                    data=data,
                    timeout=60,
                )
                if r.status_code == 201:
                    resp = r.json()
                    st.session_state.job_id = resp["job_id"]
                    st.session_state.job_status = resp["status"]
                    st.session_state.report = None
                    st.success(f"Uploaded {resp['files_uploaded']} file(s). Job: `{resp['job_id'][:8]}...`")
                    st.rerun()
                else:
                    st.error(f"Upload failed: {r.status_code} - {r.text}")
            except requests.ConnectionError:
                st.error("Cannot connect to API. Is the backend running?")

# === STATUS TAB ===
with status_tab:
    st.subheader("Job Status")

    job_id_input = st.text_input(
        "Job ID",
        value=st.session_state.job_id or "",
        placeholder="Enter job ID or upload first",
    )

    if st.button("Refresh Status"):
        if job_id_input:
            try:
                r = requests.get(f"{api_url}/documents/{job_id_input}", timeout=10)
                if r.status_code == 200:
                    data = r.json()
                    st.session_state.job_id = data["job_id"]
                    st.session_state.job_status = data["status"]

                    status_color = {
                        "completed": "green",
                        "failed": "red",
                        "queued": "orange",
                        "processing": "blue",
                    }.get(data["status"], "gray")

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Status", data["status"])
                    col2.metric("Files", len(data.get("files", [])))
                    col3.metric("Collection", data.get("collection") or "default")

                    st.json(data)
                elif r.status_code == 404:
                    st.warning("Job not found")
                else:
                    st.error(f"Error: {r.status_code}")
            except requests.ConnectionError:
                st.error("Cannot connect to API")
        else:
            st.warning("Enter a job ID first")

    if st.session_state.job_id:
        st.info(f"Current job: `{st.session_state.job_id[:8]}...` (status: {st.session_state.job_status})")

# === REPORT TAB ===
with report_tab:
    st.subheader("Analysis Report")

    report_job_id = st.text_input(
        "Job ID for report",
        value=st.session_state.job_id or "",
        placeholder="Job must be completed first",
        key="report_job_id",
    )

    if st.button("Load Report"):
        if report_job_id:
            try:
                r = requests.get(f"{api_url}/documents/{report_job_id}/report", timeout=10)
                if r.status_code == 200:
                    data = r.json()
                    st.session_state.report = data.get("report", {})
                    if not st.session_state.report:
                        st.info("No report available yet. The job may still be processing.")
                else:
                    st.error(f"Error: {r.status_code} - {r.text}")
            except requests.ConnectionError:
                st.error("Cannot connect to API")
        else:
            st.warning("Enter a job ID")

    if st.session_state.report:
        report = st.session_state.report

        if "compliance_score" in report:
            score = report["compliance_score"]
            st.metric("Compliance Score", f"{score}/100")

        if "obligations" in report and report["obligations"]:
            st.markdown("**Obligations**")
            for i, obligation in enumerate(report["obligations"], 1):
                if isinstance(obligation, dict):
                    st.markdown(f"{i}. {obligation.get('text', obligation)}")
                else:
                    st.markdown(f"{i}. {obligation}")

        if "entities" in report and report["entities"]:
            st.markdown("**Entities**")
            for entity in report["entities"]:
                if isinstance(entity, dict):
                    label = entity.get("label", "Entity")
                    text = entity.get("text", str(entity))
                    st.markdown(f"- **{label}**: {text}")
                else:
                    st.markdown(f"- {entity}")

        if "risky_terms" in report and report["risky_terms"]:
            st.markdown("**Risky / Ambiguous Terms**")
            for term in report["risky_terms"]:
                if isinstance(term, dict):
                    st.warning(f"{term.get('clause', term)}")
                else:
                    st.warning(str(term))

        with st.expander("Full Report JSON"):
            st.json(report)

# === QUERY TAB ===
with query_tab:
    st.subheader("Ask a Question (RAG)")

    query_job_id = st.text_input(
        "Job ID to query",
        value=st.session_state.job_id or "",
        placeholder="Job must be completed first",
        key="query_job_id",
    )

    query_text = st.text_area(
        "Your question",
        placeholder="e.g. What are the termination clauses?",
        height=100,
    )

    top_k = st.slider("Context chunks (top_k)", min_value=1, max_value=20, value=5)

    if st.button("Ask", type="primary", disabled=not query_text):
        if query_job_id:
            with st.spinner("Thinking..."):
                try:
                    r = requests.post(
                        f"{api_url}/query/{query_job_id}",
                        json={"query": query_text, "top_k": top_k},
                        timeout=120,
                    )
                    if r.status_code == 200:
                        data = r.json()
                        st.markdown("### Answer")
                        st.markdown(data["answer"])

                        if data.get("context_chunks"):
                            with st.expander("Source Chunks"):
                                for chunk in data["context_chunks"]:
                                    st.markdown(f"**Chunk {chunk['index']}** (score: {chunk.get('score', 'N/A'):.3f})")
                                    st.text(chunk["content"][:500])
                                    st.divider()
                    else:
                        st.error(f"Query failed: {r.status_code} - {r.text}")
                except requests.ConnectionError:
                    st.error("Cannot connect to API")
        else:
            st.warning("Enter a job ID first")
