from llama_index.graph_stores.neo4j import Neo4jGraphStore
from llama_index.core import StorageContext
from llama_index.core import Settings
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

# Load text data
text = "data.txt"
loader = TextLoader(text)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
texts = text_splitter.split_documents(documents)

from langchain_community.llms import Ollama
from langchain_experimental.graph_transformers.llm import LLMGraphTransformer
import getpass
import os
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    KnowledgeGraphIndex,
)

import logging
import sys

from IPython.display import Markdown, display

logging.basicConfig(
    stream=sys.stdout, level=logging.INFO
)  # logging.DEBUG for more verbose output
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))



# Initialize LLM
llm = Ollama(model="mistral",temperature=0)

embed_model = HuggingFaceEmbedding(model_name='mixedbread-ai/mxbai-embed-large-v1')
Settings.llm = llm
Settings.embed_model = embed_model

# Extract Knowledge Graph
llm_transformer = LLMGraphTransformer(llm=llm)
graph_documents = llm_transformer.convert_to_graph_documents(texts)

documents = SimpleDirectoryReader(
    "./data"
).load_data()

# Store Knowledge Graph in Neo4j
graph_store = Neo4jGraphStore(url="bolt://localhost:7687", username="neo4j", password="admin001", database="neo4j")
# graph_store.write_graph(graph_documents)
storage_context = StorageContext.from_defaults(graph_store=graph_store)

index = KnowledgeGraphIndex.from_documents(
    documents,
    storage_context=storage_context,
    max_triplets_per_chunk=2,
)



query_engine = index.as_query_engine(
    include_text=False, response_mode="tree_summarize"
)

response = query_engine.query("Tell me more about prismaticAI")

display(Markdown(f"<b>{response}</b>"))

