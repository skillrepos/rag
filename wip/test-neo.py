from langchain_community.graphs import Neo4jGraph

url = "bolt://localhost:7687"
username ="neo4j"
password = "admin001"

graph = Neo4jGraph(
    url=url, 
    username=username, 
    password=password
)
