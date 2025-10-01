from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_community.embeddings import OllamaEmbeddings, HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings


def create_embedding_model(model_type="huggingface", model_name=None):
    if model_type == "ollama":
        model_name = model_name or "nomic-embed-text"
        print(f"Loading Ollama embedding model: {model_name}")
        return OllamaEmbeddings(model=model_name)
    elif model_type == "huggingface":
        model_name = model_name or "all-MiniLM-L6-v2"
        print(f"Loading HuggingFace embedding model: {model_name}")
        return HuggingFaceEmbeddings(model_name=model_name)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def create_llm(model_name="phi3:mini", temperature=0):
    print(f"Loading LLM: {model_name} with temperature={temperature}")
    llm = OllamaLLM(model=model_name, temperature=temperature)
    return llm


def create_compression_retriever(vector_store, base_k=10, top_n=4):
    print(f"Creating compression retriever with base_k={base_k}, top_n={top_n}")
    base_retriever = vector_store.as_retriever(search_kwargs={"k": base_k})
    reranker_model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    compressor = CrossEncoderReranker(model=reranker_model, top_n=top_n)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    print("Compression retriever created successfully.")
    return compression_retriever


def create_rag_chain(llm, retriever, chain_type="stuff"):
    print(f"Creating RAG chain with chain_type={chain_type}")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=True
    )
    print("RAG chain created successfully.")
    return qa_chain


def query_rag(qa_chain, question):
    print(f"Processing question: {question[:50]}...")
    response = qa_chain.invoke({"query": question})
    return response