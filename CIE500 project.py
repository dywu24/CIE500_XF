import pandas as pd
import shapely.wkt
from sklearn.neighbors import NearestNeighbors
import torch
from torch_geometric.data import Data
import numpy as np

# Load Excel file
df = pd.read_excel("AADT_with_snow.xlsx")

# Drop rows with missing key fields
df = df.dropna(subset=["AADT", "sample_1", "wkt_geom"])

# Extract coordinates from WKT
coords = df["wkt_geom"].apply(shapely.wkt.loads).apply(lambda p: (p.x, p.y))
coord_array = np.array(coords.tolist())

# 🚧 Step 3: Construct edge_index based on nearest neighbors
nn = NearestNeighbors(n_neighbors=4, algorithm="ball_tree").fit(coord_array)
distances, indices = nn.kneighbors(coord_array)

edges = []
for i, nbrs in enumerate(indices):
    for j in nbrs[1:]:  # skip self
        edges.append((i, j))

edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

# 📊 Step 4: Build feature matrix x and label y
features = df[["sample_1"]].values  # snow depth only for now
x = torch.tensor(features, dtype=torch.float)
y = torch.tensor(df["AADT"].values, dtype=torch.float)

# 🧱 Step 5: Create GCN-ready data object
data = Data(x=x, edge_index=edge_index, y=y)
print(data)

# 🧠 Step 6: Define GCN model (regression)
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch.nn import Linear

class GCNRegression(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.lin = Linear(hidden_channels, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.lin(x)
        return x.squeeze()  # regression

# 📐 Step 7: Train the model
from torch_geometric.data import DataLoader

model = GCNRegression(in_channels=x.shape[1], hidden_channels=16)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()

model.train()
for epoch in range(100):
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out, data.y)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1:03d}, Loss: {loss.item():.4f}")

# 📈 Step 8: Visualize prediction vs actual
import matplotlib.pyplot as plt

model.eval()
pred = model(data).detach().numpy()
true = data.y.numpy()

plt.figure(figsize=(6,6))
plt.scatter(true, pred, alpha=0.6)
plt.xlabel("True AADT")
plt.ylabel("Predicted AADT")
plt.title("GCN Predicted vs Actual AADT")
plt.grid(True)
plt.plot([true.min(), true.max()], [true.min(), true.max()], 'r--')
plt.show()