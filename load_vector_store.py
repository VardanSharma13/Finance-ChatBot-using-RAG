# ### After the vector store is saved, load the database.
# This database takes 2 arguments.
#  - Path to tht database
#  - Embedding Model

# from create_vector_store import hf_embeddings_model
# from create_vector_store import vector_store_data_path
# from langchain_community.vectorstores import Chroma
# importing like above is not a good idea  :)

import os
os.environ['HF_HOME'] = "./hf_cache"
from langchain_community.vectorstores import Chroma
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

vector_store_data_path = "chroma_data"
embedding_model_name = "intfloat/multilingual-e5-large-instruct"

hf_embeddings_model = HuggingFaceEmbeddings(model_name=embedding_model_name)





vector_database_db = Chroma(
    persist_directory=vector_store_data_path,
    embedding_function=hf_embeddings_model
)
