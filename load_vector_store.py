# ### After the vector store is saved, load the database.
# This database takes 2 arguments.
#  - Path to tht database
#  - Embedding Model

vector_database_db = Chroma(
    persist_directory=vector_store_data_path,
    embedding_function=hf_embeddings_model
)
