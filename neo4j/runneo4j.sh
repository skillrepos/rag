docker run \
     --name neo4j \
     -p7474:7474 -p7687:7687 \
     --env NEO4J_dbms_directories_import="/" \
     --env NEO4JLABS_PLUGINS='["apoc"]' \
     --env NEO4J_AUTH=neo4j/neo4jtest \
     neo4j:custom