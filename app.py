import streamlit as st
import os
import tempfile
from dotenv import load_dotenv

load_dotenv()

from src.data_processing import process_pdf_to_chunks
from src.vector_store import (
    build_vector_store, 
    load_vector_store, 
    generate_collection_name,
    vector_store_exists
)
from src.rag_components import (
    create_embedding_model,
    create_llm,
    create_compression_retriever,
    create_rag_chain,
    query_rag
)

st.set_page_config(
    page_title="RIZA - RAG Chat",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-title {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-top: 0;
        margin-bottom: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "current_doc" not in st.session_state:
    st.session_state.current_doc = None
if "collection_name" not in st.session_state:
    st.session_state.collection_name = None
if "persist_dir" not in st.session_state:
    st.session_state.persist_dir = None

with st.sidebar:
    st.title("⚙️ Configuration")
    
    st.subheader("Embedding Model")
    embedding_type = st.selectbox(
        "Model Type",
        ["huggingface", "ollama"],
        help="Choose embedding model type"
    )
    
    if embedding_type == "ollama":
        embedding_model_name = st.text_input(
            "Model Name",
            "nomic-embed-text",
            help="Ollama embedding model name"
        )
    else:
        embedding_model_name = st.text_input(
            "Model Name",
            "all-MiniLM-L6-v2",
            help="HuggingFace embedding model name"
        )
    
    st.divider()
    
    st.subheader("LLM Model")
    llm_type = st.selectbox(
        "LLM Provider",
        ["Ollama", "OpenAI API", "Google API"],
        help="Choose LLM provider"
    )
    
    llm_type_map = {
        "Ollama": "ollama",
        "OpenAI API": "openai",
        "Google API": "google"
    }
    llm_type_key = llm_type_map[llm_type]
    
    if llm_type == "Ollama":
        llm_model = st.text_input(
            "Model Name",
            "phi3:mini",
            help="Ollama model name for generation"
        )
    elif llm_type == "OpenAI API":
        llm_model = st.selectbox(
            "Model Name",
            ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o", "gpt-4o-mini"],
            help="OpenAI model for generation"
        )
        if not os.getenv("OPENAI_API_KEY"):
            st.warning("⚠️ OPENAI_API_KEY not found in .env file")
    elif llm_type == "Google API":
        llm_model = st.selectbox(
            "Model Name",
            ["gemini-2.5-pro", "gemini-2.5-flash", "gemini-2.5-flash-lite"],
            help="Google Gemini model for generation"
        )
        if not os.getenv("GOOGLE_API_KEY"):
            st.warning("⚠️ GOOGLE_API_KEY not found in .env file")
    
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.1,
        help="Controls response creativity"
    )
    
    st.divider()
    
    st.subheader("Retrieval Settings")
    base_k = st.slider(
        "Base K (documents to retrieve)",
        min_value=5,
        max_value=20,
        value=10,
        help="Number of documents before reranking"
    )
    
    top_n = st.slider(
        "Top N (after reranking)",
        min_value=1,
        max_value=10,
        value=4,
        help="Final number of documents after reranking"
    )
    
    st.divider()
    
    st.subheader("📄 Document Upload")
    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        help="Upload a PDF to chat with"
    )
    
    if uploaded_file is not None:
        if st.button("Process Document", type="primary", use_container_width=True):
            with st.spinner("Processing document..."):
                try:
                    collection_name = generate_collection_name(uploaded_file.name)
                    persist_dir = "./data/processed/chroma_db"
                    
                    if vector_store_exists(persist_dir, collection_name):
                        st.info("📦 Existing vector store found. Loading...")
                        embedding_model = create_embedding_model(embedding_type, embedding_model_name)
                        vector_store = load_vector_store(embedding_model, persist_dir, collection_name)
                    else:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            tmp_path = tmp_file.name
                        
                        st.info("📄 Converting PDF to chunks...")
                        chunks = process_pdf_to_chunks(tmp_path)
                        
                        st.info("🔤 Loading embedding model...")
                        embedding_model = create_embedding_model(embedding_type, embedding_model_name)
                        
                        st.info("🗄️ Building vector store...")
                        vector_store = build_vector_store(chunks, embedding_model, persist_dir, collection_name)
                        
                        os.unlink(tmp_path)
                    
                    st.info(f"🤖 Setting up LLM ({llm_type})...")
                    llm = create_llm(llm_type_key, llm_model, temperature)
                    
                    st.info("🔗 Creating compression retriever...")
                    retriever = create_compression_retriever(vector_store, base_k, top_n)
                    
                    st.info("⚡ Creating RAG chain...")
                    qa_chain = create_rag_chain(llm, retriever)
                    
                    st.session_state.vector_store = vector_store
                    st.session_state.qa_chain = qa_chain
                    st.session_state.current_doc = uploaded_file.name
                    st.session_state.collection_name = collection_name
                    st.session_state.persist_dir = persist_dir
                    st.session_state.messages = []
                    
                    st.success(f"✅ Document '{uploaded_file.name}' ready!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error processing document: {str(e)}")
                    import traceback
                    st.error(traceback.format_exc())
    
    if st.session_state.current_doc:
        st.divider()
        st.success(f"📄 **Active Document:**\n{st.session_state.current_doc}")
        st.caption(f"🔖 Collection: {st.session_state.collection_name}")
        
        if st.button("🗑️ Clear Session", use_container_width=True):
            st.session_state.vector_store = None
            st.session_state.qa_chain = None
            st.session_state.current_doc = None
            st.session_state.collection_name = None
            st.session_state.persist_dir = None
            st.session_state.messages = []
            st.rerun()

st.markdown('<h1 class="main-title">RIZA</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">A simple RAG chat interface</p>', unsafe_allow_html=True)

if not st.session_state.current_doc:
    st.info("👈 Upload a PDF document in the sidebar to start!")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📤 1. Upload")
        st.write("Submit a PDF file through the sidebar")
    with col2:
        st.markdown("### ⚙️ 2. Configure")
        st.write("Adjust models and parameters as needed")
    with col3:
        st.markdown("### 💬 3. Chat")
        st.write("Ask questions about the document content")
else:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("📚 View sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}:**")
                        st.text(source[:300] + "..." if len(source) > 300 else source)
                        if i < len(message["sources"]):
                            st.divider()
    
    if prompt := st.chat_input("Ask a question about the document..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = query_rag(st.session_state.qa_chain, prompt)
                    answer = response["result"]
                    source_docs = response.get("source_documents", [])
                    
                    st.markdown(answer)
                    
                    sources = [doc.page_content for doc in source_docs]
                    
                    if sources:
                        with st.expander("📚 View sources"):
                            for i, source in enumerate(sources, 1):
                                st.markdown(f"**Source {i}:**")
                                st.text(source[:300] + "..." if len(source) > 300 else source)
                                if i < len(sources):
                                    st.divider()
                    
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })
                    
                except Exception as e:
                    error_msg = f"❌ Error generating response: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

if st.session_state.current_doc:
    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.caption(f"💬 Messages: {len(st.session_state.messages)}")
    with col2:
        st.caption(f"🤖 LLM: {llm_type}")
    with col3:
        st.caption(f"🔤 Embedding: {embedding_type}")
    with col4:
        st.caption(f"📊 Retrieval: {base_k}→{top_n}")