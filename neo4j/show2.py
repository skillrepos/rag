from neo4j import GraphDatabase
from pyvis.network import Network

# Connect to your Neo4j instance
uri = "bolt://localhost:7687"
driver = GraphDatabase.driver(uri, auth=("neo4j", "neo4jtest"))

# Sample Cypher query to retrieve nodes and relationships
cypher_query = """
MATCH (n)-[r]->(m)
RETURN n, r, m
LIMIT 100
"""

# Execute the query and fetch the results

with driver.session() as session:
  result = session.run(cypher_query)

  net = Network(cdn_resources = "remote", directed = True, height = '500px',width = '100%', notebook = True)

  for record in result:
    node_a = record["n"]
    node_b = record["m"]
    relationship = record["r"]

    #add nodes
    net.add_node(node_a.element_id)
    net.add_node(node_ab.element_id)
    net.add_edge(node_a.element_id)

#save html format
net.show("example_file.html",notebook=False)