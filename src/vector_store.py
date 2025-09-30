from langchain_community.vectorstores import Chroma


def build_vector_store(
    chunks, 
    embedding_model, 
    persist_directory):
  
    print(f"Criando Vector Store com {len(chunks)} chunks em '{persist_directory}'...")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory
    )
    print("Vector Store criado com sucesso.")
    return db


def load_vector_store(
    embedding_model, 
    persist_directory
):
    print(f"Carregando Vector Store de '{persist_directory}'...")
    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model
    )
    print("Vector Store carregado com sucesso.")
    return db