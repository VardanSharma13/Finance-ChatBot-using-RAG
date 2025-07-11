# Now making the UI
# We will use gradio to provide the chat UI


from RAG_pipline_using_LangChain import conversation_chain



import gradio as gr

def chat(question, history):
    result = conversation_chain.invoke({
        "question": question,
    })['answer'].split("<|end_header_id|>")[-1].strip()
    return result


gr.ChatInterface(chat,
                title="💰 Finance Assistant",
                description = "Ask me anything about finance, investing, stocks, or economic concepts!",
                examples = [
                          "What is the difference between a stock and a bond?",
                          "How does compound interest work?",
                          "What are the key indicators to evaluate a company's financial health?",
                          "Explain the concept of risk vs return.",
                          "What is the time value of money?"
                            ],
                theme="default",
                type="messages",
                ).queue().launch(debug=True)