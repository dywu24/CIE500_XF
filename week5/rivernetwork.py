#%%
import geopandas as gpd
import networkx as nx
import momepy
import matplotlib.pyplot as pit

river_data = gpd.read_file(
    "/Users/dwu24/Desktop/CIE500Fan/week5/rivernetwork/networkoregon.shp"
)

river_data.plot()
print(river_data.head)

river_data_exploded = river_data.explode(ignore_index=True, index_parts=False)[
    ["geometry"]
]

river_data_exploded.plot()

print(river_data_exploded.head())

G = momepy.gdf_to_nx(
    river_data_exploded,
    approach="primal",
    multigraph=False,
    directed=False,
    length="length"
)
G.remove_edges_from(nx.selfloop_edges(G))

pos = {node: node for node in G.nodes()}