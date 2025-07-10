

# Models and Dataset dependencies
!pip install transformers datasets accelerate bitsandbytes fsspec==2023.9.2
# Langchain Dependencies
!pip install langchain langchain-community langchain-text-splitters langchain-huggingface langchain-core
# Other Dependencies
!pip install chromadb unstructured gradio






import os
os.environ['HF_HOME'] = "./hf_cache"

# Dataset
from datasets import load_dataset

ds = load_dataset("alvanlii/finance-textbooks")

ds

print(ds['train']['book_title'][0])


for row in ds['train']:
  print(row)
  break

from tqdm import tqdm

def save_books_with_title(ds, base_path):
  book_titles = {}
  for row in tqdm(ds['train']):
    book_title = row['book_title']
    book_text = row['book_text']
    if book_title not in book_titles:
      book_titles[book_title] = 0
    filename = os.path.join(base_path, f"{book_title}.txt") if book_titles[book_title] == 0 else os.path.join(base_path, f"{book_title}_{book_titles[book_title]}.txt")
    with open(filename, 'w') as f:
      f.write(book_text)


!mkdir books

save_books_with_title(ds, './books')

# Create Vector Database
# - ## Model
#  - To calculate Embeddings for both sentence and data.
# - ## Vector Store
#  - To store vectors and help in search


# Load document loader
from langchain_community.document_loaders import DirectoryLoader

# Load text splitter.
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.schema.document import Document

# Import the vector database.
from langchain_community.vectorstores import Chroma

# Load the model to create embeddings.
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from tqdm import tqdm


# Load the data

# from the directory load all the .txt files.
loader = DirectoryLoader("./books", glob="*.txt", show_progress=True, use_multithreading=True)
docs = loader.load()
len(docs)



# Splitting the data to create Chunks for better retrieval (Mini-Documents)
# - For eg:
#     - If a question has answer in 5 different documents so without splitting retreiver will give all 5 documents of 800 lines (total 4000 lines)
#     - We wanted to avoid this because searching --> LLM and Prompt length also increases

# - When we split we create Mini-Document with same meta information to the original document.
#     - Author name, Book title, etc.


# - We split smarty.
#     - We have some overlap: some of the previous and next context is preserved
#         - This helps in better retrieval because information loss is avoided due to sudden break

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

chunked_docs = text_splitter.split_documents(docs)

print(len(chunked_docs))


#Embedding Model

embedding_model_name = "intfloat/multilingual-e5-large-instruct"

hf_embeddings_model = HuggingFaceEmbeddings(model_name=embedding_model_name)



# ## Create Vector Database
#  - Passing Model
#  - And Documents (splitted as chunks)

#  ### Note
#  As this is using a model, it will require a GPU to create embeddings otherwise in CPU it will be very Slow.


import time
vector_store_data_path = "chroma_data"

print("Starting")
start_time = time.time()

vectorstore = Chroma.from_documents(
    chunked_docs , hf_embeddings_model, persist_directory= vector_store_data_path,
)

end_time = time.time()
print("Done in {} minutes".format((end_time - start_time) / 60))


