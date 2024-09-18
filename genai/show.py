import pandas as pd
import numpy as np
import networkx as nx
from IPython.display import display, HTML

G = nx.Graph()
rels = [["Fred", "George"],
    ["Harry", "Rita"],
    ["Fred", "Ginny"],
    ["Tom", "Ginny"],
    ["Harry", "Ginny"],
    ["Harry", "George"],
    ["Frank", "Ginny"],
    ["Marge", "Rita"],
    ["Fred", "Rita"]]

G.add_edges_from(rels)

from pyvis.network import Network
net = Network(notebook=True)
net.from_nx(G)
#net = Network()

net.from_nx(G)

net.save_graph("networkx-pyvis.html")
HTML(filename="networkx-pyvis.html")