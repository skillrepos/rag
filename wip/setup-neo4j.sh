# Install Neo4j
pip install py2neo
pip install langchain
pip install langchain_community
pip install neo4j

wget -O - https://debian.neo4j.com/neotechnology.gpg.key | sudo apt-key add -
sudo echo 'deb https://debian.neo4j.com stable 5' | sudo tee /etc/apt/sources.list.d/neo4j.list
sudo apt-get update
sudo apt-get install neo4j
sudo mv /var/lib/neo4j/labs/apoc-5.23.0.1-core.jar /var/lib/neo4j/plugins/
sudo echo dbms.security.procedures.unrestricted=algo.*,apoc.* >> /etc/neo4j/neo4j.conf
neo4j-admin set-initial-password admin

sudo service neo4j start
sudo service neo4j status
