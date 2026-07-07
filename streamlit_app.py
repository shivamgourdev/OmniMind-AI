import requests
import streamlit as st

BACKEND_URL = "http://127.0.0.1:8000"
REQUEST_TIMEOUT_UPLOAD = 120  
REQUEST_TIMEOUT_ASK = 60

st.set_page_config(
    page_title="OmniMind AI",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 OmniMind AI")
st.caption("Advanced Multi-Agent Hybrid RAG System")

st.divider()

st.header("1. Upload Documents")

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True,
    help="You can select multiple PDFs at once. Re-uploading a file with "
         "the same name replaces its previous content."
)

upload_clicked = st.button("Upload PDFs", type="primary", disabled=not uploaded_files)

if upload_clicked:

    files_data = [
        ("files", (file.name, file.getvalue(), "application/pdf"))
        for file in uploaded_files
    ]

    with st.spinner(f"Processing {len(uploaded_files)} file(s) — chunking, embedding, and indexing..."):

        try:
            response = requests.post(
                f"{BACKEND_URL}/upload",
                files=files_data,
                timeout=REQUEST_TIMEOUT_UPLOAD
            )

        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the backend. Please check the server is running and reachable.")

        except requests.exceptions.Timeout:
            st.error("The upload timed out. Try uploading fewer or smaller files at once.")

        except Exception as e:
            st.error(f"Unexpected error: {e}")

        else:

            if response.status_code == 200:

                data = response.json()
                results = data.get("uploaded_files", [])

                if data.get("success"):
                    st.success(data.get("message", "Upload complete."))
                else:
                    st.error(data.get("message", "Upload failed."))

                if results:
                    for item in results:

                        if item["status"] == "success":
                            st.write(f"✅ **{item['filename']}** — {item['chunks']} chunk(s) indexed")
                        else:
                            reason = item.get("reason") or "Unknown error."
                            st.write(f"❌ **{item['filename']}** — {reason}")

            else:
                st.error(f"Upload failed (status {response.status_code})")
                st.code(response.text)

st.divider()



st.header("2. Ask a Question")

with st.form("ask_form", clear_on_submit=False):

    question = st.text_input(
        "Ask your question",
        placeholder="e.g. Summarize ShivamGour.pdf, or Compare A.pdf and B.pdf"
    )

    ask_clicked = st.form_submit_button("Ask AI", type="primary")

if ask_clicked:

    if not question or not question.strip():
        st.warning("Please enter a question.")

    else:

        with st.spinner("Thinking..."):

            try:
                response = requests.post(
                    f"{BACKEND_URL}/ask",
                    json={"question": question},
                    timeout=REQUEST_TIMEOUT_ASK
                )

            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the backend. Please check the server is running and reachable.")

            except requests.exceptions.Timeout:
                st.error("The request timed out. Please try again.")

            except Exception as e:
                st.error(f"Unexpected error: {e}")

            else:

                if response.status_code == 200:

                    data = response.json()

                    missing_files = data.get("missing_files") or []
                    if missing_files:
                        st.warning(
                            "Not found in the uploaded documents: " + ", ".join(missing_files)
                        )

                    st.subheader("Answer")
                    st.write(data.get("answer", ""))

                    col1, col2 = st.columns(2)

                    with col1:
                        st.subheader("Searched Files")
                        searched_files = data.get("searched_files") or []

                        if searched_files:
                            for file in searched_files:
                                st.write(f"• {file}")
                        else:
                            st.caption("No files were searched.")

                    with col2:
                        st.subheader("Relevant Sources")
                        sources = data.get("sources") or []

                        if sources:
                            for source in sources:
                                st.write(f"• {source}")
                        else:
                            st.caption("No sources returned.")

                elif response.status_code == 422:
                    st.error("Invalid request.")
                    st.code(response.text)

                else:
                    st.error(f"Request failed (status {response.status_code})")
                    st.code(response.text)
