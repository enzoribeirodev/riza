from langchain.chains import RetrievalQA
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


def create_compression_retriever(vector_store, base_k=10, top_n=4):
    base_retriever = vector_store.as_retriever(search_kwargs={"k": base_k})
    
    reranker_model = HuggingFaceCrossEncoder(model_name="cross-encoder/ms-marco-MiniLM-L-6-v2")
    
    compressor = CrossEncoderReranker(model=reranker_model, top_n=top_n)
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=base_retriever
    )
    
    return compression_retriever


def create_rag_chain(llm, retriever, chain_type="stuff"):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain