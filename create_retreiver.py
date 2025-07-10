# ### Now we have the database and we want a retriever.
# We will make our retriever from the database and return top 4 matching documents (`chunks` in our case)

# https://python.langchain.com/docs/integrations/retrievers/self_query/chroma_self_query/

retreiver = vector_database_db.as_retriever(k=4)

for ret_doc in retreiver.invoke("What are stocks?"):
  print("*"*50)
  print(ret_doc)
  print("*"*50)
  break
