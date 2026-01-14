from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import HypotheticalDocumentEmbedder
import torch
from groq import Groq

import os
from dotenv import load_dotenv
load_dotenv()


## for increased efficiency
os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.set_num_threads(4)


groq_api_key = os.getenv("GROQ_API_KEY")

##############################################################################################

hf_embeddings = HuggingFaceEmbeddings(
    model_name = "BAAI/bge-small-en-v1.5",
    encode_kwargs = {'normalize_embeddings':True},
)   

llm = ChatGroq(model="openai/gpt-oss-20b", 
               groq_api_key=groq_api_key)

llm_summarize = ChatGroq(model="llama-3.1-70b-versatile", 
               groq_api_key=groq_api_key)


vision_model = "meta-llama/llama-4-scout-17b-16e-instruct"
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

##############################################################################################


hf_reranker_encoder = "cross-encoder/ms-marco-MiniLM-L-6-v2"

##############################################################################################

hyde_base_embedding =  HuggingFaceEmbeddings(
    model_name = "BAAI/bge-small-en-v1.5",
    encode_kwargs = {'normalize_embeddings':True},
)

hyde_embedding = HypotheticalDocumentEmbedder.from_llm(llm = llm, 
                                              base_embeddings = hyde_base_embedding,
                                              prompt_key="sci_fact")




