docker run -d \
  --name neo4j -p 7474:7474 -p 7687:7687 -v /workspaces/rag/certs:/var/lib/neo4j/certificates \
  -e NEO4J_server_https_enabled=true \
  -e NEO4J_dbms_ssl_policy_https_enabled=true \
  -e NEO4J_dbms_ssl_policy_https_base__directory=certificates/https \
  -e NEO4J_dbms_ssl_policy_https_private__key=neo4j.key \
  -e NEO4J_dbms_ssl_policy_https_public__certificate=neo4j.cert \
  -e NEO4J_server_bolt_tls__level=REQUIRED \
  -e NEO4J_dbms_ssl_policy_bolt_enabled=true \
  -e NEO4J_dbms_ssl_policy_bolt_base__directory=certificates/bolt \
  -e NEO4J_dbms_ssl_policy_bolt_private__key=neo4j.key \
  -e NEO4J_dbms_ssl_policy_bolt_public__certificate=neo4j.cert \
  -e NEO4J_dbms_ssl_policy_https_trust_all=true \
  -e NEO4J_dbms_ssl_policy_bolt_trust_all=true \
  neo4j:latest
