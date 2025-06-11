import os
import tempfile

os.environ["CHROMA_DISABLE_TELEMETRY"] = "True"

import streamlit as st
import torch
from PyPDF2 import PdfReader
# ↓ dùng loader có sẵn trong langchain core
from langchain.document_loaders import PyPDFLoader  
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    pipeline,
)
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline

# —————————————— State init ——————————————
if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None
if "model_loaded" not in st.session_state:
    st.session_state.model_loaded = False
if "embeddings" not in st.session_state:
    st.session_state.embeddings = None
if "llm" not in st.session_state:
    st.session_state.llm = None

# —————————————— Load embeddings ——————————————
@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(model_name="bkai-foundation-models/vietnamese-bi-encoder")

# —————————————— Load LLM ——————————————
@st.cache_resource
def load_llm():
    # NF4 4-bit quant config
    nf4 = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_dtype=False,
    )
    MODEL = "lmsys/vicuna-7b-v1.5"
    model = AutoModelForCausalLM.from_pretrained(
        MODEL,
        quantization_config=nf4,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=512,
        pad_token_id=tokenizer.eos_token_id,
        device_map="auto",
    )
    return HuggingFacePipeline(pipeline=pipe)

# —————————————— Xử lý PDF → RAG chain ——————————————
def process_pdf(uploaded_file):
    # 1) Lưu tạm PDF
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        pdf_path = tmp.name

    # 2) Load & split văn bản
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = splitter.split_documents(docs)

    # 3) Build FAISS index
    embeddings = st.session_state.embeddings
    vector_db = FAISS.from_documents(chunks, embeddings)

    # 4) Chuẩn bị prompt template
    prompt_template = """Bạn là trợ lý thông minh. Dưới đây là phần context được trích xuất từ PDF:
{context}

Hãy trả lời câu hỏi sau bằng tiếng Việt, nếu không có trong context thì xin lỗi và nói bạn không biết:
Question: {question}
Answer:"""
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )

    # 5) Tạo một function đơn giản cho RAG
    def rag_chain(question: str) -> str:
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        docs = retriever.get_relevant_documents(question)
        context = "\n\n".join([d.page_content for d in docs])
        text_in = prompt.format(context=context, question=question)
        return st.session_state.llm(text_in)

    # 6) Cleanup
    os.unlink(pdf_path)
    return rag_chain, len(chunks)

# —————————————— Giao diện ——————————————
st.set_page_config(page_title="PDF RAG Assistant", layout="wide")
st.title("📝 PDF RAG Assistant (Tiếng Việt)")
st.markdown("""
Ứng dụng RAG giúp bạn hỏi đáp trực tiếp trong PDF, chỉ với vài bước đơn giản:

1. **Upload** file PDF.
2. **Process** để tách và index.
3. **Ask** câu hỏi, nhận trả lời ngay!
""")

# —————————————— Load model lần đầu ——————————————
if not st.session_state.model_loaded:
    with st.spinner("Đang tải model và embeddings…"):
        st.session_state.embeddings = load_embeddings()
        st.session_state.llm = load_llm()
        st.session_state.model_loaded = True
    st.success("Model và embeddings đã sẵn sàng!")
    st.experimental_rerun()

# —————————————— Upload & Process ——————————————
uploaded = st.file_uploader("Chọn file PDF", type=["pdf"])
if uploaded and st.button("Process PDF"):
    with st.spinner("Đang xử lý PDF…"):
        st.session_state.rag_chain, n_chunks = process_pdf(uploaded)
    st.success(f"PDF đã được xử lý và chia thành {n_chunks} chunks.")

# —————————————— Hỏi đáp ——————————————
if st.session_state.rag_chain:
    q = st.text_input("Nhập câu hỏi của bạn:")
    if q:
        with st.spinner("Đang sinh câu trả lời…"):
            resp = st.session_state.rag_chain(q)
        st.markdown("**Answer:**")
        st.write(resp)
