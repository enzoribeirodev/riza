import os
import hashlib
from langchain_community.vectorstores import Chroma

def generate_collection_name(filename):
    hash_object = hashlib.md5(filename.encode())
    hash_hex = hash_object.hexdigest()[:8]
    clean_name = "".join(c if c.isalnum() else "_" for c in filename)
    clean_name = clean_name[:30]
    collection_name = f"rag_{clean_name}_{hash_hex}"
    return collection_name

def build_vector_store(chunks, embedding_model, persist_directory, collection_name=None):
    print(f"Creating Vector Store with {len(chunks)} chunks in '{persist_directory}'...")
    os.makedirs(persist_directory, exist_ok=True)
    
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_name=collection_name
    )
    
    print(f"Vector Store created successfully. Collection: {collection_name}")
    return db

def load_vector_store(embedding_model, persist_directory, collection_name):
    print(f"Loading Vector Store from '{persist_directory}' (collection: {collection_name})...")
    
    if not os.path.exists(persist_directory):
        raise ValueError(f"Directory '{persist_directory}' does not exist.")
    
    db = Chroma(
        persist_directory=persist_directory,
        embedding_function=embedding_model,
        collection_name=collection_name
    )
    
    print("Vector Store loaded successfully.")
    return db

def vector_store_exists(persist_directory, collection_name):
    if not os.path.exists(persist_directory):
        return False
    chroma_file = os.path.join(persist_directory, "chroma.sqlite3")
    return os.path.exists(chroma_file)