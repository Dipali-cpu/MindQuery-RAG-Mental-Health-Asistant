import streamlit as st
from rag_pipeline import process_pdf_and_query
import tempfile
import os

st.set_page_config(
    page_title="MindQuery",
    page_icon="🧠",
    layout="centered"
)

st.markdown("""
    <h1 style='color: #1D9E75;'>🧠 MindQuery</h1>
    <p style='color: gray;'>AI-powered Mental Health Research Assistant</p>
    <p style='color: gray; font-size: 13px;'>Built with LangChain · FAISS · HuggingFace · Groq LLaMA 3.3</p>
    <hr>
""", unsafe_allow_html=True)

st.subheader("Upload your research PDF")
uploaded_file = st.file_uploader(
    "Choose a PDF file",
    type="pdf"
)

if uploaded_file is not None:
    st.success(f"Uploaded: {uploaded_file.name}")
    
    question = st.text_input(
        "Ask a question about your document",
        placeholder="e.g. What % of tech workers sought mental health treatment?"
    )

if st.button("Ask MindQuery"):
    if question:
        with st.spinner("Searching through your document..."):

            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name

            try:
                result = process_pdf_and_query(tmp_path, question)

            except Exception as e:
                import traceback
                st.error(str(e))
                st.code(traceback.format_exc())
                st.stop()

            os.unlink(tmp_path)

        st.markdown("### Answer")
        st.write(result["answer"])

        st.markdown("### Sources Used")
        for i, doc in enumerate(result["sources"]):
            with st.expander(
                f"Source {i+1} — Page {doc.metadata.get('page', 'N/A') + 1}"
            ):
                st.write(doc.page_content)

    else:
        st.warning("Please enter a question!")







            