import networkx as nx
import random
from itertools import combinations
import matplotlib.pyplot as plt
import numpy as np

## I would like to generate 9 random nodes in the horizontal space (0,3) and vertical space (0,3)
random.seed(69)
pos = {i: (random.random(), random.random()) for i in range(9)}

# Generate a random edge list
edge_list = []
for u, v in combinations(range(9), 2):  # Generate all possible edges
    if random.random() < 0.3:  # Add an edge with a 30% probability
        edge_list.append((u, v))

# Print to debug
print(edge_list)  # Should contain tuples like [(0, 2), (1, 3), ...]

# Create graph from edge list
G = nx.Graph()
G.add_edges_from(edge_list)

# Draw the graph
fig, ax = plt.subplots()
nx.draw_networkx(G, pos=pos, with_labels=True, ax=ax, node_color='skyblue', edge_color='gray', node_size=500)
plt.show()
plt.tight_layout()
ax.set_aspect("equal")  # set the equal scale of horizontal and vertical
ax.axis("off")  # remove the frame of the generated figure
plt.savefig(
    "/Users/dwu24/Desktop/CIE500Fan/class1/classexamplegraph.jpg",
    dpi=300,
    bbox_inches="tight",
)


def bmatrix(
    a,
):  # reference source: https://stackoverflow.com/questions/17129290/numpy-2d-and-1d-array-to-latex-bmatrix
    """Returns a LaTeX bmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if len(a.shape) > 3:
        raise ValueError("bmatrix can at most display two dimensions")
    lines = str(a).replace("[", "").replace("]", "").splitlines()
    rv = [r"\begin{bmatrix}"]
    rv += ["  " + " & ".join(l.split()) + r"\\" for l in lines]
    rv += [r"\end{bmatrix}"]
    return "\n".join(rv)

A = nx.adjacency_matrix(G).toarray()

print(f"The latex version of adjacency matrix is \n {bmatrix(A)}")


# Now let's add a self-loop to the network

G.add_edge(6, 6)
G.add_edge(3, 3)


fig, ax = plt.subplots()
nx.draw_networkx(G, pos=pos, with_labels=True, ax=ax, arrowstyle="<|-", style="dashed")
plt.tight_layout()
ax.set_aspect("equal")  # set the equal scale of horizontal and vertical
ax.axis("off")  # remove the frame of the generated figure
plt.savefig(
    "/Users/dwu24/Desktop/CIE500Fan/class1/examplegraph_selfloop.jpg",
    dpi=300,
    bbox_inches="tight",
)
# Finally, we can get the edgelist and adjancency matrix from Graph directly.

print(f"The adjancency matrix of G is \n {nx.adjacency_matrix(G).toarray()}")

print(f"The edge list of G is \n {nx.to_edgelist(G)}")