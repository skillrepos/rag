# Install Neo4j
wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -
echo 'deb https://debian.neo4j.com stable 4.2' | sudo tee /etc/apt/sources.list.d/neo4j.list
sudo apt-get update
sudo apt-get install neo4j
mv /var/lib/neo4j/labs/apoc-4.2.0.11-core.jar /var/lib/neo4j/plugins/
echo dbms.security.procedures.unrestricted=algo.*,apoc.* >> /etc/neo4j/neo4j.conf
neo4j-admin set-initial-password admin
pip install py2neo
pip install langchain
pip install langchain_community
pip install neo4j

service neo4j start
service neo4j status
