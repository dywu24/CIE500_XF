#%%

import networkx as nx
import matplotlib.pyplot as plt
import random
import numpy as np
import itertools


def edge_list(node_num, probability):
    edge_list_created = []
    node_list = list(range(node_num))
    for node_pair in itertools.combinations(node_list,2):
        p = random.random()
        if p < probability:
            edge_list_created.append(node_pair)
    return edge_list_created

edgelist = edge_list(33, 0.6)
G = nx.from_edgelist(edgelist)


pos = nx.spring_layout(G)

# nx.draw_networkx(G,pos,with_labels=False,node_size=9)


import geopandas as gpd
import momepy

gpd_data = gpd.read_file("/Users/dwu24/Desktop/CIE500Fan/week5/rivernetwork/networkoregon.shp")

gpd_data.head()

gpd_data_exploded = gpd_data.explode(ignore_index=True, index_parts=False)

G = momepy.gdf_to_nx(
gpd_data_exploded,
approach="primal",
multigraph=False,
directed=False,
length="length"
)
G.remove_edges_from(nx.selfloop_edges(G))

pos = {node: node for node in G.nodes()}

from swmmio import Model
import networkx as nx

model = Model('Drainage_Example.inp')
G = model.network
pos = {}
for node in G.nodes():
    pos[node] = G.nodes[node]['geometry']['coordinates']

nx.draw_networkx(G,pos)
plt.plot()

import wntr

wn = wntr.network.WaterNetworkModel("Drinking_example.inp")

G =wntr.network.io.to_graph(wn)

pos = {}
for node in G.nodes():
    pos[node]  = G.nodes[node]['pos']

# nx.draw_networkx(G,
#     pos=pos,
#     width=0.5,
#     with_labels=False,
#     node_color="lightblue",
#     edge_color="gray",
#     arrowsize = 0.5,
#     node_size=0,)
