from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from llama_index.graph_stores.neo4j import Neo4jGraphStore

# Load text data

loader = TextLoader("file6.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20)
texts = text_splitter.split_documents(documents)

from langchain_community.llms import Ollama
from langchain_experimental.graph_transformers import LLMGraphTransformer
import getpass
import os


# Initialize LLM
llm = Ollama(model="llama3")

# Extract Knowledge Graph
llm_transformer = LLMGraphTransformer(llm=llm)
graph_documents = llm_transformer.convert_to_graph_documents(texts)
 
#from langchain.graph_stores import Neo4jGraphStore



# Store Knowledge Graph in Neo4j
graph_store = Neo4jGraphStore(url="neo4j://localhost:7687", username="neo4j", password="neo4jtest")
graph_store.write_graph(graph_documents)

from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import KnowledgeGraphRAGRetriever
from llama_index.core.response_synthesis import ResponseSynthesizer

# Retrieve Knowledge for RAG
graph_rag_retriever = KnowledgeGraphRAGRetriever(storage_context=graph_store.storage_context, verbose=True)
query_engine = RetrieverQueryEngine.from_args(graph_rag_retriever)

def query_and_synthesize(query):
    retrieved_context = query_engine.query(query)
    response = response_synthesizer.synthesize(query, retrieved_context)
    print(f"Query: {query}")
    print(f"Answer: {response}\n")

# Initialize the ResponseSynthesizer instance
response_synthesizer = ResponseSynthesizer(llm)

# Query 1
query_and_synthesize("Where does Sarah work?")

# Query 2
query_and_synthesize("Who works for prismaticAI?")

# Query 3
query_and_synthesize("Does Michael work for the same company as Sarah?")
