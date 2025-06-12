%%writefile app_rag.py

import streamlit as st
import tempfile
import os
import torch
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_experimental.text_splitter import SemanticChunker
from langchain_chroma import Chroma
from langchain_huggingface.llms import HuggingFacePipeline
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "llm" not in st.session_state:
    st.session_state.llm = None

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder")

@st.cache_resource
def load_llm():
    nf4_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_quant=True,
        bnb_4bit_use_double_dtype=torch.bfloat16
    )

    MODEL_NAME = "lmsys/vicuna-7b-v1.5"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=nf4_config,
        low_cpu_mem_usage=True
    )
    tokenizer= AutoTokenizer.from_pretrained(MODEL_NAME)
    model_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id,
        device_map="auto"
    )
    return HuggingFacePipeline(pipeline=model_pipeline)

def process_pdf(upload_file):
    with tempfile.NamedTemporaryFile(delete=False) as temp_file:
        temp_file.write(upload_file.getvalue())
        temp_file_path = temp_file.name
    loader = PyPDFLoader(file_path=temp_file_path)
    documents = loader.load()


    semantic_splitter =  SemanticChunker(
        embeddings = st.session_state.embeddings,
        buffer_size=1,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=95,
        min_chunk_size=500,
        add_start_index=True
    )

    docs = semantic_splitter.split_documents(documents)
    vector_db = Chroma.from_documents(
        documents = docs,
        embedding = st.session_state.embeddings
    )
    retriever = vector_db.as_retriever()
    prompt = hub.pull("rlm/rag-prompt")

    def format_docs(docs):
      return "\n\n".join(doc.page_content for doc in docs)

    rag_chain = (
        {
            "context": retriever | format_docs, "question": RunnablePassthrough()
        }
        | prompt
        | st.session_state.llm
        | StrOutputParser()
    )

    os.unlink(temp_file_path)
    return rag_chain, len(docs)

# Xây dựng giao diện người dùng
st.set_page_config(page_title="PDF RAG Assistant", layout="wide")
st.title("PDF RAG Assistant")

st.markdown("""
** Ứng dụng AI giúp bạn hỏi đáp trực tiếp với nội dung tài liệu PDF bằng tiếng Việt**
\n** Cách sử dụng đơn giản**
1. ** Upload file ** (Chọn file PDF từ máy tính và nhấn "Xử lý PDF")
2. ** Đặt câu hỏi ** (Nhập câu hỏi về nội dung tài liệu và nhận câu trả lời ngay lập tức)
""")

if not st.session_state.model_loaded:
    st.info("Model loading...")
    st.session_state.embeddings = load_embeddings()
    st.session_state.llm = load_llm()
    st.session_state.model_loaded = True
    st.success("Model loaded successfully!")
    st.rerun()

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])
if uploaded_file and st.button("Process PDF"):
    with st.spinner("Processing PDF..."):
        st.session_state.rag_chain, num_chunks = process_pdf(uploaded_file)
        st.success("PDF processed successfully!")

if st.session_state.rag_chain:
    question = st.text_input("Ask a question about the PDF:")
    if question:
        with st.spinner("Generating answer..."):
            output = st.session_state.rag_chain.invoke(question)
            answer = output.split('Answer: ')[1].strip() if "Answer: " in output else output.strip()
            st.write("**Answer:**")
            st.write(answer)

