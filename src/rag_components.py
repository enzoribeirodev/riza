from langchain.chains import RetrievalQA

def create_rag_chain(llm, retriever, chain_type="stuff"):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=True
    )
    return qa_chain