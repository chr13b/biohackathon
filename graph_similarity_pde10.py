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

PDB_DIR   = "pdb_pde10"
TRAIN_DIR = "apherisfold_inputs/apherisfold_inputs/train"
VAL_DIR   = "apherisfold_inputs/apherisfold_inputs/val"
CONTACT_CUTOFF = 8.0
N_SPECTRAL = 10


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
        break
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
    from scipy.spatial.distance import cdist
    n = len(coords)
    G = nx.Graph()
    G.add_nodes_from(range(n))
    chunk_size = 128
    upper_dists = []
    for i in range(0, n, chunk_size):
        block = cdist(coords[i:i+chunk_size], coords)
        for li, row in enumerate(block):
            gi = i + li
            for j in range(gi + 1, n):
                d = row[j]
                if 0 < d < cutoff:
                    G.add_edge(gi, j, weight=float(d))
            upper_dists.extend(row[gi+1:].tolist())
    return G, np.array(upper_dists)


def spectral_features(G, k=N_SPECTRAL):
    L = nx.normalized_laplacian_matrix(G).toarray().astype(float)
    vals = np.linalg.eigvalsh(L)   # deterministic, sorted ascending
    return vals[:k] if len(vals) >= k else np.pad(vals, (0, k - len(vals)))


def graph_features(G, upper_dists):
    degrees = np.array([d for _, d in G.degree()])
    cc = nx.average_clustering(G)
    spec = spectral_features(G)
    feats = np.concatenate([
        [G.number_of_nodes(), G.number_of_edges(), nx.density(G)],
        [degrees.mean(), degrees.std(), degrees.max()],
        [cc],
        [upper_dists.mean(), upper_dists.std()],
        spec,
    ])
    return feats


def load_structures(directory, label, verbose=True):
    paths = sorted(glob.glob(os.path.join(directory, "*.cif")))
    names, features, labels = [], [], []
    for i, path in enumerate(paths):
        name = os.path.basename(path).replace(".cif", "")
        try:
            coords = get_ca_coords(path)
            if len(coords) < 5:
                continue
            G, dists = build_contact_graph(coords)
            feats = graph_features(G, dists)
            names.append(name)
            features.append(feats)
            labels.append(label)
            if verbose:
                print(f"  [{i+1}/{len(paths)}] {name}: {len(coords)} residues, {G.number_of_edges()} contacts")
        except Exception as e:
            if verbose:
                print(f"  Error on {name}: {e}")
    return names, np.array(features), labels


print("=== Loading PDB PDE10 structures ===")
pdb_names, pdb_feats, pdb_labels = load_structures(PDB_DIR, "PDB", verbose=False)
print(f"  Loaded {len(pdb_names)} structures")

print("\n=== Loading train structures ===")
train_names, train_feats, train_labels = load_structures(TRAIN_DIR, "train")

print("\n=== Loading val structures ===")
val_names, val_feats, val_labels = load_structures(VAL_DIR, "val")

all_names  = pdb_names  + train_names  + val_names
all_feats  = np.vstack([pdb_feats, train_feats, val_feats])
all_labels = pdb_labels + train_labels + val_labels
n_pdb, n_train, n_val = len(pdb_names), len(train_names), len(val_names)
n = len(all_names)
print(f"\nTotal: {n} structures  ({n_pdb} PDB, {n_train} train, {n_val} val)")

# Normalise
scaler = StandardScaler()
feats_scaled = scaler.fit_transform(all_feats)

# Pairwise distance
pairwise_dist = squareform(pdist(feats_scaled, metric="euclidean"))
similarity    = 1 / (1 + pairwise_dist)

# UMAP
print("Running UMAP...")
reducer = umap.UMAP(n_components=2, metric="euclidean",
                    n_neighbors=min(15, n - 1), random_state=42)
embedding = reducer.fit_transform(feats_scaled)

# Hierarchical clustering order (for heatmap)
Z     = linkage(squareform(pairwise_dist), method="average")
order = leaves_list(Z)
sim_reordered    = similarity[np.ix_(order, order)]
labels_reordered = [all_labels[i] for i in order]

COLOR_MAP  = {"PDB": "#cbd5e0", "train": "#2b6cb0", "val": "#c05621"}
MARKER_MAP = {"PDB": "o",      "train": "o",        "val": "^"}

# Residue counts per structure (from node count feature, index 0)
res_counts = all_feats[:, 0]

# ── figure layout ────────────────────────────────────────────────────────────
plt.style.use("seaborn-v0_8-whitegrid")
fig = plt.figure(figsize=(24, 10))
gs  = fig.add_gridspec(1, 2, width_ratios=[1.5, 1], wspace=0.28)

# ── Panel 1 : annotated UMAP ─────────────────────────────────────────────────
ax = fig.add_subplot(gs[0])

# density contour for PDB background
pdb_mask = [i for i, l in enumerate(all_labels) if l == "PDB"]
try:
    from scipy.stats import gaussian_kde
    xy  = embedding[pdb_mask].T
    kde = gaussian_kde(xy, bw_method=0.25)
    xg  = np.linspace(embedding[:, 0].min() - 1, embedding[:, 0].max() + 1, 150)
    yg  = np.linspace(embedding[:, 1].min() - 1, embedding[:, 1].max() + 1, 150)
    XX, YY = np.meshgrid(xg, yg)
    ZZ = kde(np.vstack([XX.ravel(), YY.ravel()])).reshape(XX.shape)
    ax.contourf(XX, YY, ZZ, levels=8, cmap="Blues", alpha=0.25, zorder=1)
    ax.contour( XX, YY, ZZ, levels=8, colors="#90cdf4", linewidths=0.5,
                alpha=0.5, zorder=1)
except Exception:
    pass

# PDB scatter (small, behind)
ax.scatter(embedding[pdb_mask, 0], embedding[pdb_mask, 1],
           c=COLOR_MAP["PDB"], s=40, alpha=0.8, zorder=2,
           edgecolors="#718096", linewidths=0.4,
           label=f"PDB structures (n={n_pdb})")

# Label PDB points with their IDs
for i in pdb_mask:
    ax.annotate(all_names[i], (embedding[i, 0], embedding[i, 1]),
                fontsize=4, color="#718096", alpha=0.7,
                ha="center", va="bottom",
                xytext=(0, 3), textcoords="offset points")

# Train / Val scatter (prominent, in front)
for label, marker, zorder in [("train", "o", 5), ("val", "^", 5)]:
    mask = [i for i, l in enumerate(all_labels) if l == label]
    ax.scatter(embedding[mask, 0], embedding[mask, 1],
               c=COLOR_MAP[label], s=140, marker=marker,
               edgecolors="white", linewidths=1.2,
               zorder=zorder,
               label=f"{'Training' if label=='train' else 'Validation'} set (n={len(mask)})")
    for i in mask:
        ax.annotate(
            all_names[i],
            (embedding[i, 0], embedding[i, 1]),
            fontsize=7.5, fontweight="bold",
            color=COLOR_MAP[label],
            ha="center", va="bottom",
            xytext=(0, 7), textcoords="offset points",
        )

# Cluster annotation ellipses
from matplotlib.patches import Ellipse
monomer_idx = [i for i in range(n) if res_counts[i] < 500]
dimer_idx   = [i for i in range(n) if res_counts[i] >= 500]
for idx, label_text, color in [
    (monomer_idx, "Monomers\n(~310 residues)",  "#e53e3e"),
    (dimer_idx,   "Dimers\n(~1255 residues)",   "#276749"),
]:
    if not idx:
        continue
    xs, ys = embedding[idx, 0], embedding[idx, 1]
    cx, cy = xs.mean(), ys.mean()
    w, h   = (xs.max() - xs.min()) * 1.25 + 1.5, (ys.max() - ys.min()) * 1.25 + 1.5
    ell = Ellipse((cx, cy), width=w, height=h,
                  edgecolor=color, facecolor="none",
                  linestyle="--", linewidth=1.8, zorder=6)
    ax.add_patch(ell)
    ax.text(cx, cy + h / 2 + 0.3, label_text,
            ha="center", va="bottom", fontsize=9,
            color=color, fontweight="bold", zorder=7)

ax.set_title("Structural Landscape of PDE10A Structures\n(Contact-graph features · UMAP projection)",
             fontsize=13, pad=12)
ax.set_xlabel("UMAP dimension 1", fontsize=11)
ax.set_ylabel("UMAP dimension 2", fontsize=11)
ax.legend(fontsize=10, framealpha=0.9, loc="best")
ax.tick_params(labelsize=9)

# ── Panel 2 : nearest-neighbour distance bar chart ───────────────────────────
ax2 = fig.add_subplot(gs[1])

tv_names, tv_splits, tv_nn_dists, tv_nn_names = [], [], [], []
for i, name in enumerate(train_names + val_names):
    split = "train" if i < n_train else "val"
    row   = pairwise_dist[n_pdb + i, :n_pdb]
    j     = row.argmin()
    tv_names.append(name)
    tv_splits.append(split)
    tv_nn_dists.append(row[j])
    tv_nn_names.append(pdb_names[j])

# Sort by distance ascending
sort_order  = np.argsort(tv_nn_dists)
sorted_names   = [tv_names[i]    for i in sort_order]
sorted_splits  = [tv_splits[i]   for i in sort_order]
sorted_dists   = [tv_nn_dists[i] for i in sort_order]
sorted_nn      = [tv_nn_names[i] for i in sort_order]

bar_colors = [COLOR_MAP[s] for s in sorted_splits]
y_pos = np.arange(len(sorted_names))

bars = ax2.barh(y_pos, sorted_dists, color=bar_colors,
                edgecolor="white", linewidth=0.5, height=0.7)

# Annotate nearest PDB neighbour name
for yi, (d, nn) in enumerate(zip(sorted_dists, sorted_nn)):
    ax2.text(d + 0.02, yi, nn, va="center", fontsize=7, color="#4a5568")

ax2.set_yticks(y_pos)
ax2.set_yticklabels(sorted_names, fontsize=8)
for tick, split in zip(ax2.get_yticklabels(), sorted_splits):
    tick.set_color(COLOR_MAP[split])

# Threshold line at 0.5 (close neighbours)
ax2.axvline(0.5, color="#e53e3e", linestyle="--", linewidth=1.2, zorder=5)
ax2.text(0.52, len(sorted_names) - 0.5, "similarity\nthreshold",
         color="#e53e3e", fontsize=8, va="top")

ax2.set_xlabel("Distance to nearest PDB neighbour\n(graph feature space)", fontsize=11)
ax2.set_title("Dataset Coverage\nHow close each train/val structure is to the PDB set",
              fontsize=12, pad=12)

train_patch = mpatches.Patch(color=COLOR_MAP["train"], label="Training set")
val_patch   = mpatches.Patch(color=COLOR_MAP["val"],   label="Validation set")
ax2.legend(handles=[train_patch, val_patch], fontsize=10, loc="lower right")
ax2.tick_params(labelsize=9)
ax2.set_xlim(0, max(sorted_dists) * 1.35)
ax2.invert_yaxis()

plt.suptitle("PDE10A Structure Graph Similarity Analysis",
             fontsize=15, fontweight="bold", y=1.01)

out = "graph_similarity_pde10.png"
plt.savefig(out, dpi=180, bbox_inches="tight")
print(f"Saved: {out}")

# Summary stats
tv_dist = pairwise_dist[n_pdb:n_pdb+n_train, n_pdb+n_train:]
pdb_train_dist = pairwise_dist[:n_pdb, n_pdb:n_pdb+n_train]
pdb_val_dist   = pairwise_dist[:n_pdb, n_pdb+n_train:]

print("\nDistance summary (feature space):")
print(f"  PDB-PDB    mean: {pairwise_dist[:n_pdb, :n_pdb].mean():.3f}")
print(f"  Train-val  mean: {tv_dist.mean():.3f}  min: {tv_dist.min():.3f}")
print(f"  PDB-train  mean: {pdb_train_dist.mean():.3f}  min: {pdb_train_dist.min():.3f}")
print(f"  PDB-val    mean: {pdb_val_dist.mean():.3f}  min: {pdb_val_dist.min():.3f}")

# Nearest PDB neighbour for each train/val structure
print("\nNearest PDB neighbour for each train/val structure:")
for i, name in enumerate(train_names + val_names):
    split = "train" if i < n_train else "val"
    row = pairwise_dist[n_pdb + i, :n_pdb]
    j = row.argmin()
    print(f"  {name} ({split}) -> {pdb_names[j]}  dist={row[j]:.3f}")
