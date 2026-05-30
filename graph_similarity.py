import os
import glob
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.spatial.distance import pdist, squareform
from scipy.sparse.linalg import eigsh
from scipy.cluster.hierarchy import linkage, dendrogram, leaves_list
from sklearn.preprocessing import StandardScaler
import umap

try:
    import gemmi
    PARSER = "gemmi"
except ImportError:
    from Bio.PDB import MMCIFParser
    PARSER = "biopython"

TRAIN_DIR = "apherisfold_inputs/apherisfold_inputs/train"
VAL_DIR = "apherisfold_inputs/apherisfold_inputs/val"
CONTACT_CUTOFF = 8.0   # Angstroms, Cα-Cα
N_SPECTRAL = 10        # top eigenvalues of graph Laplacian


def get_ca_coords_gemmi(cif_path):
    st = gemmi.read_structure(cif_path)
    coords = []
    for model in st:
        for chain in model:
            for residue in chain:
                for atom in residue:
                    if atom.name == "CA":
                        pos = atom.pos
                        coords.append([pos.x, pos.y, pos.z])
        break  # first model only
    return np.array(coords)


def get_ca_coords_biopython(cif_path):
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("s", cif_path)
    coords = []
    for model in structure:
        for chain in model:
            for residue in chain:
                if "CA" in residue:
                    coords.append(residue["CA"].get_vector().get_array())
        break
    return np.array(coords)


def get_ca_coords(cif_path):
    if PARSER == "gemmi":
        return get_ca_coords_gemmi(cif_path)
    return get_ca_coords_biopython(cif_path)


def build_contact_graph(coords, cutoff=CONTACT_CUTOFF):
    n = len(coords)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    diff = coords[:, None, :] - coords[None, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=-1))
    rows, cols = np.where((dists < cutoff) & (dists > 0))
    for i, j in zip(rows, cols):
        if i < j:
            G.add_edge(i, j, weight=float(dists[i, j]))
    return G, dists


def spectral_features(G, k=N_SPECTRAL):
    n = G.number_of_nodes()
    L = nx.normalized_laplacian_matrix(G).astype(float)
    k_actual = min(k, n - 2)
    if k_actual < 1:
        return np.zeros(k)
    vals = eigsh(L, k=k_actual, which="SM", return_eigenvectors=False)
    vals = np.sort(vals)
    # Pad if graph is smaller than k
    if len(vals) < k:
        vals = np.pad(vals, (0, k - len(vals)), constant_values=0)
    return vals


def graph_features(G, dists):
    degrees = np.array([d for _, d in G.degree()])
    cc = nx.average_clustering(G)
    spec = spectral_features(G)
    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()
    density = nx.density(G)
    # Contact map statistics
    upper = dists[np.triu_indices(len(dists), k=1)]
    mean_dist = upper.mean()
    std_dist = upper.std()

    feats = np.concatenate([
        [n_nodes, n_edges, density],
        [degrees.mean(), degrees.std(), degrees.max()],
        [cc],
        [mean_dist, std_dist],
        spec,
    ])
    return feats


def load_structures(directory):
    paths = sorted(glob.glob(os.path.join(directory, "*.cif")))
    names, features = [], []
    for path in paths:
        name = os.path.basename(path).replace(".cif", "")
        try:
            coords = get_ca_coords(path)
            if len(coords) < 5:
                print(f"  Skipping {name}: too few CA atoms ({len(coords)})")
                continue
            G, dists = build_contact_graph(coords)
            feats = graph_features(G, dists)
            names.append(name)
            features.append(feats)
            print(f"  {name}: {len(coords)} residues, {G.number_of_edges()} contacts")
        except Exception as e:
            print(f"  Error on {name}: {e}")
    return names, np.array(features)


print("=== Loading train structures ===")
train_names, train_feats = load_structures(TRAIN_DIR)
print("\n=== Loading val structures ===")
val_names, val_feats = load_structures(VAL_DIR)

all_names = [f"{n} (train)" for n in train_names] + [f"{n} (val)" for n in val_names]
all_feats = np.vstack([train_feats, val_feats])
is_train = [True] * len(train_names) + [False] * len(val_names)
n_train = len(train_names)
n = len(all_names)

print(f"\nTotal: {n} structures ({n_train} train, {n - n_train} val)")

# Normalise
scaler = StandardScaler()
all_feats_scaled = scaler.fit_transform(all_feats)

# Pairwise Euclidean distance in feature space
pairwise_dist = squareform(pdist(all_feats_scaled, metric="euclidean"))
similarity = 1 / (1 + pairwise_dist)

# UMAP embedding
reducer = umap.UMAP(n_components=2, metric="euclidean",
                    n_neighbors=min(5, n - 1), random_state=42)
embedding = reducer.fit_transform(all_feats_scaled)

# Hierarchical clustering order for heatmap
Z = linkage(squareform(pairwise_dist), method="average")
order = leaves_list(Z)
sim_reordered = similarity[np.ix_(order, order)]
names_reordered = [all_names[i] for i in order]
is_train_reordered = [is_train[i] for i in order]

colors = ["#1f77b4" if t else "#ff7f0e" for t in is_train]

fig = plt.figure(figsize=(20, 8))
gs = fig.add_gridspec(1, 3, width_ratios=[1.2, 0.05, 1.5], wspace=0.35)

# --- UMAP scatter ---
ax_umap = fig.add_subplot(gs[0])
for i, (x, y) in enumerate(embedding):
    ax_umap.scatter(x, y, color=colors[i], s=80, zorder=3)
    ax_umap.annotate(all_names[i].split(" ")[0], (x, y),
                     fontsize=6, ha="center", va="bottom", xytext=(0, 4),
                     textcoords="offset points")
ax_umap.set_title("UMAP — graph feature space", fontsize=11)
ax_umap.set_xlabel("UMAP 1")
ax_umap.set_ylabel("UMAP 2")
train_patch = mpatches.Patch(color="#1f77b4", label="Train")
val_patch = mpatches.Patch(color="#ff7f0e", label="Val")
ax_umap.legend(handles=[train_patch, val_patch], fontsize=8)

# --- Heatmap ---
ax_heat = fig.add_subplot(gs[2])
im = ax_heat.imshow(sim_reordered, aspect="auto", cmap="viridis", vmin=0, vmax=1)
ax_heat.set_xticks(range(n))
ax_heat.set_xticklabels(names_reordered, rotation=90, fontsize=6)
ax_heat.set_yticks(range(n))
ax_heat.set_yticklabels(names_reordered, fontsize=6)
for i, (lx, ly, is_tr) in enumerate(
        zip(ax_heat.get_xticklabels(), ax_heat.get_yticklabels(), is_train_reordered)):
    c = "#1f77b4" if is_tr else "#ff7f0e"
    lx.set_color(c)
    ly.set_color(c)
plt.colorbar(im, ax=ax_heat, label="Similarity")
ax_heat.set_title("Pairwise graph similarity (hierarchical order)", fontsize=11)

plt.suptitle("Protein Structure Graph Similarity — ApherisFold Dataset", fontsize=13)
out = "graph_similarity.png"
plt.savefig(out, dpi=150, bbox_inches="tight")
print(f"\nSaved: {out}")

# Summary
print("\nFeature-space distance summary:")
tv = pairwise_dist[:n_train, n_train:]
print(f"  Train-train mean dist: {pairwise_dist[:n_train, :n_train].mean():.3f}")
print(f"  Val-val   mean dist:   {pairwise_dist[n_train:, n_train:].mean():.3f}")
print(f"  Train-val mean dist:   {tv.mean():.3f}")
print(f"  Train-val min  dist:   {tv.min():.3f}")
print(f"  Train-val max  dist:   {tv.max():.3f}")
