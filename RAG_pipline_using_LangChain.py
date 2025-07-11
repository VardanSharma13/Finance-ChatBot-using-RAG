## Now we will create the RAG Pipeline using `Langchain`

import os
os.environ['HF_HOME'] = "./hf_cache"


from create_retreiver import retreiver
from LLM_for_RAG_pipeline import chat_model



# Import the RAG Chain Wrapper, which will do Retrieve -> Intermediate Prompt -> LLM steps.
from langchain.chains import ConversationalRetrievalChain
# This is required to have memory to the LLM otherwise
# you have to manually pass the history every time to the LLM
from langchain.memory import ConversationBufferWindowMemory
# For printing intermediate retrival chunks
from langchain_core.callbacks import StdOutCallbackHandler
# This is to give a system prompt to the LLM to define the behaviour
from langchain_core.messages import SystemMessage



# Set up the conversation memory for the chat
# We are using sliding window to only keep most recent messages.
memory = ConversationBufferWindowMemory(memory_key='chat_history', return_messages=True,  k=3)





# Set up the system message to guide the model's behavior

system_message = SystemMessage(
    content="""You are a Finance assistant trained to help people in India.
    - Use Indian context (₹, Indian companies, tax rules, etc.) wherever applicable.
    - Provide responses only within the finance domain.
    - Reply in markdown format with sufficient elaboration.
    - For questions outside finance, reply: "Sorry! not trained for out of domain questions :)" and end the response.
"""
)



# putting it together: set up the conversation chain with the LLM, the vector store and memory

conversation_chain = ConversationalRetrievalChain.from_llm(llm=chat_model, retriever=retreiver, memory=memory)

## Uncomment the line below if you want to see the output of the chain in the console
# conversation_chain = ConversationalRetrievalChain.from_llm(llm=chat_model, retriever=retreiver, memory=memory, callbacks=[StdOutCallbackHandler()])

# Insert the system message to the top of the message prompts.
conversation_chain.combine_docs_chain.llm_chain.prompt.messages.insert(0, system_message)

## Uncomment the line below if you want to see the system prompt messages used by the chain
# print(conversation_chain.combine_docs_chain.llm_chain.prompt.messages)