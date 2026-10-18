#!/usr/bin/env python3
"""
Export the Axonal Labs website brain (fsaverage5 pial surface, from brainmesh.js)
into a compact binary for the FocusLoop iOS app, with Yeo-7 network labels.

Verified facts this script relies on:
  * brainmesh.js holds the fsaverage5 pial surface (10,242 vertices per hemisphere,
    20,484 total), left hemisphere then right, in the ORIGINAL FreeSurfer vertex
    order. Faces are identical to nilearn's fsaverage5, and positions correlate
    r = 1.00 with RAS after the axis swap (site x = R, y = S, z = A).
  * Because the vertex order is preserved, any model output defined on fsaverage5
    (per-vertex predictions) maps 1:1 onto this mesh: vertex i == fsaverage5 vertex i.

Labels: Schaefer et al. 2018, 400 parcels / 7 networks, fsaverage5 (CBIG, MIT
license). Each parcel belongs to one Yeo-7 network.

Output (little-endian) BrainMesh.bin:
  char[4]  magic "FLBM"
  uint32   version (1)
  uint32   nVerts
  uint32   nIndices
  float32  positions[nVerts*3]      site coordinates (x=R, y=S, z=A), |z|<=1.3
  float32  normals[nVerts*3]        outward unit normals
  uint8    ao[nVerts]               baked ambient occlusion (from the website)
  uint8    network[nVerts]          0 = medial wall, 1..7 = NETWORKS[i-1]
  uint16   parcel[nVerts]           0 = medial wall, 1..400 (LH 1-200, RH 201-400)
  uint8    netWeights[nVerts*8]     soft network membership * 255 (slot 7 unused)
  uint16   indices[nIndices]        triangle list

Plus BrainAtlas.json: network names/colors, parcel names, provenance.

Usage:
  python export_brain_mesh.py --mesh path/to/brainmesh.js --out Resources/Brain
"""
import argparse, base64, json, os, re, struct, sys, urllib.request
import numpy as np

NETWORKS = [  # Yeo 2011 7-network order, Schaefer short names
    ("Vis", "Visual"),
    ("SomMot", "Somatomotor"),
    ("DorsAttn", "Dorsal Attention"),
    ("SalVentAttn", "Salience / Ventral Attention"),
    ("Limbic", "Limbic"),
    ("Cont", "Frontoparietal Control"),
    ("Default", "Default Mode"),
]
YEO_COLORS = [  # canonical Yeo-7 colors
    [120, 18, 134], [70, 130, 180], [0, 118, 14], [196, 58, 250],
    [220, 248, 164], [230, 148, 34], [205, 62, 78],
]
SCHAEFER_URL = ("https://raw.githubusercontent.com/ThomasYeoLab/CBIG/master/stable_projects/"
                "brain_parcellation/Schaefer2018_LocalGlobal/Parcellations/FreeSurfer5.3/"
                "fsaverage5/label/{h}.Schaefer2018_400Parcels_7Networks_order.annot")


def decode_site_mesh(path):
    src = open(path, encoding="utf-8").read()
    n = int(re.search(r"\bn:(\d+)", src).group(1))
    s = float(re.search(r"\bs:([\d.]+)", src).group(1))
    field = lambda k: base64.b64decode(re.search(r"\b" + k + r':"([^"]+)"', src).group(1))
    pos = np.frombuffer(field("pos"), np.int16).astype(np.float32).reshape(-1, 3) * (s / 32767.0)
    idx = np.frombuffer(field("idx"), np.uint16).astype(np.int64).reshape(-1, 3)
    ao = np.frombuffer(field("ao"), np.uint8).copy()
    assert pos.shape[0] == n == ao.shape[0] == 20484, "expected fsaverage5 (20484 vertices)"
    return pos, idx, ao


def outward_normals(pos, idx):
    """Same method as brain.js: area-weighted face normals, flipped to point outward."""
    a, b, c = pos[idx[:, 0]], pos[idx[:, 1]], pos[idx[:, 2]]
    fn = np.cross(b - a, c - a)
    nrm = np.zeros_like(pos)
    for k in range(3):
        np.add.at(nrm, idx[:, k], fn)
    if np.sum(np.einsum("ij,ij->i", nrm, pos) < 0) > len(pos) / 2:
        nrm = -nrm
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True).clip(1e-9)
    return nrm.astype(np.float32)


def load_schaefer(annot_dir):
    import nibabel as nib
    net_short = [s for s, _ in NETWORKS]
    networks, parcels, parcel_names = [], [], []
    for hi, h in enumerate(["lh", "rh"]):
        f = os.path.join(annot_dir, f"{h}.Schaefer400_7.annot")
        if not os.path.exists(f):
            print(f"downloading {h} Schaefer-400 annotation ...", file=sys.stderr)
            urllib.request.urlretrieve(SCHAEFER_URL.format(h=h), f)
        lab, _, names = nib.freesurfer.read_annot(f)
        names = [x.decode() for x in names]
        if hi == 0:
            parcel_names.extend(names[1:])          # 200 LH parcels
        else:
            parcel_names.extend(names[1:])          # 200 RH parcels
        lab = lab.clip(0)
        net = np.zeros(len(lab), np.uint8)
        for li, nm in enumerate(names):
            if li == 0:
                continue
            net[lab == li] = net_short.index(nm.split("_")[2]) + 1
        networks.append(net)
        parcels.append(np.where(lab > 0, lab + hi * 200, 0).astype(np.uint16))
    return np.concatenate(networks), np.concatenate(parcels), parcel_names


def soft_weights(network, idx, iters=4):
    """One-hot networks smoothed over the mesh graph -> soft edges that glow like the site."""
    n = len(network)
    W = np.zeros((n, 8), np.float32)
    has = network > 0
    W[np.arange(n)[has], network[has] - 1] = 1.0
    # vertex adjacency (undirected)
    e = np.vstack([idx[:, [0, 1]], idx[:, [1, 2]], idx[:, [2, 0]]])
    e = np.vstack([e, e[:, ::-1]])
    deg = np.bincount(e[:, 0], minlength=n).astype(np.float32)
    for _ in range(iters):
        acc = np.zeros_like(W)
        np.add.at(acc, e[:, 0], W[e[:, 1]])
        W = 0.5 * W + 0.5 * acc / deg[:, None].clip(1)
    return np.round(W.clip(0, 1) * 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", required=True, help="brainmesh.js from the Axonal Labs website")
    ap.add_argument("--annots", default=os.path.dirname(os.path.abspath(__file__)),
                    help="folder holding / receiving {lh,rh}.Schaefer400_7.annot")
    ap.add_argument("--out", required=True, help="output folder (e.g. Resources/Brain)")
    ap.add_argument("--smooth", type=int, default=4)
    a = ap.parse_args()

    pos, idx, ao = decode_site_mesh(a.mesh)
    nrm = outward_normals(pos, idx)
    network, parcel, parcel_names = load_schaefer(a.annots)
    weights = soft_weights(network, idx, a.smooth)

    os.makedirs(a.out, exist_ok=True)
    flat_idx = idx.astype(np.uint16).ravel()
    with open(os.path.join(a.out, "BrainMesh.bin"), "wb") as f:
        f.write(b"FLBM" + struct.pack("<III", 1, len(pos), len(flat_idx)))
        for arr in (pos.astype("<f4"), nrm.astype("<f4"), ao, network, parcel.astype("<u2"),
                    weights, flat_idx.astype("<u2")):
            f.write(np.ascontiguousarray(arr).tobytes())

    atlas = {
        "mesh": "fsaverage5 pial (FreeSurfer), vertex order preserved; LH 0-10241, RH 10242-20483",
        "coordinates": "x = right, y = superior, z = anterior; scaled so |z| <= 1.3",
        "networks": [{"id": i + 1, "short": s, "name": nm, "color": YEO_COLORS[i]}
                     for i, (s, nm) in enumerate(NETWORKS)],
        "parcels": [{"id": i + 1, "name": nm} for i, nm in enumerate(parcel_names)],
        "sources": {
            "mesh": "Axonal Labs website brainmesh.js (fsaverage via TemplateFlow)",
            "labels": "Schaefer et al. 2018, Cerebral Cortex; 400 parcels, 7 networks (CBIG, MIT license)",
            "networks": "Yeo et al. 2011, J Neurophysiol",
        },
    }
    with open(os.path.join(a.out, "BrainAtlas.json"), "w") as f:
        json.dump(atlas, f, indent=1)

    counts = {NETWORKS[i][0]: int((network == i + 1).sum()) for i in range(7)}
    print(f"wrote {len(pos)} vertices, {len(flat_idx)//3} triangles -> {a.out}")
    print("vertices per network:", counts, "medial wall:", int((network == 0).sum()))


if __name__ == "__main__":
    main()
