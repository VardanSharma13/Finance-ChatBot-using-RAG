# # Now we will add the LLM for RAG Pipeline

# Some helping links
# - https://python.langchain.com/docs/tutorials/
# - https://python.langchain.com/docs/how_to/
# - https://python.langchain.com/docs/how_to/#qa-with-rag


from huggingface_hub import login
#login("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")

from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, BitsAndBytesConfig
import torch
from langchain_huggingface import HuggingFacePipeline, ChatHuggingFace
from transformers import pipeline



phi_4 = "microsoft/Phi-4-mini-instruct"
# Quantization Config - this allows us to load the model into memory and use less memory

quant_config = BitsAndBytesConfig(
    load_in_8bit=True,
    # bnb_8bit_compute_dtype=torch.bfloat16,
    # bnb_8bit_quant_type="nf4"
)
#

tokenizer = AutoTokenizer.from_pretrained(phi_4)
tokenizer.pad_token = tokenizer.eos_token

# model = AutoModelForCausalLM.from_pretrained(LLAMA, torch_dtype=torch.bfloat16, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(phi_4, quantization_config = quant_config, trust_remote_code=True)

print(model.get_memory_footprint())

# Define the pipeline for LLM
hf_pipeline = pipeline('text-generation', model=model, tokenizer=tokenizer, max_new_tokens=1024)

langchain_llm = HuggingFacePipeline(pipeline=hf_pipeline)

chat_model = ChatHuggingFace(llm=langchain_llm)