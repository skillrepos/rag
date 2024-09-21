from langchain.chains import GraphCypherQAChain
from langchain_community.graphs import Neo4jGraph
from langchain_community.llms import Ollama

graph = Neo4jGraph(url="bolt://localhost:7687", username="neo4j", password="neo4jtest")

print(graph.schema)
enhanced_graph = Neo4jGraph(
    url="bolt://localhost:7687",
    username="neo4j",
    password="neo4jtest",
    enhanced_schema=True,
)
print(enhanced_graph.schema)

chain = GraphCypherQAChain.from_llm(
    cypher_llm=Ollama(model="llama2",temperature=0),
    qa_llm=Ollama(model="llama2",temperature=0),
    graph=graph, verbose=True,
)

# chain.invoke({"query": "Who starred in Star Trek: Generations?"})
chain.run("""Who starred in Star Trek: Generations?""")