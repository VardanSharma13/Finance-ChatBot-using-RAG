# ### Now we have the database and we want a retriever.
# We will make our retriever from the database and return top 4 matching documents (`chunks` in our case)

# https://python.langchain.com/docs/integrations/retrievers/self_query/chroma_self_query/

import os
os.environ['HF_HOME'] = "./hf_cache"


from load_vector_store import vector_database_db

retreiver = vector_database_db.as_retriever(k=4)

# for ret_doc in retreiver.invoke("What are stocks?"):
#   print("*"*50)
#   print(ret_doc)
#   print("*"*50)
#   break
