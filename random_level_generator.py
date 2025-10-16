# random_level_generator.py
# Generates a single feasible level and writes it to "level_autogen.json".
# Connected, k-colorable by construction. Readable code, short names.

import json
import math
import random
from collections import deque, defaultdict
from pathlib import Path

# ---------- config you can tweak ----------
OUT_PATH   = "level_autogen.json"
N_NODES    = 300     # number of nodes
K_COLORS   = 3        # must be <= len(PALETTE)
EDGE_P     = 10     # cross-class edge probability (density)
PRE_FRAC   = 0.1     # fraction of nodes pre-colored (consistent)
START_MODE = "maxdeg" # "maxdeg" or "random"

PALETTE = ["Red", "Green", "Blue", "Yellow", "Purple", "Orange"]


# ---------- helpers ----------
def _labels(n):
    return [f"V{i}" for i in range(1, n + 1)]

def _avg_deg(n, edges):
    return (2.0 * len(edges) / n) if n else 0.0

def _bfs_vis(adj, s, r):
    vis = {s}
    if r <= 0:
        return vis
    q = deque([(s, 0)])
    while q:
        v, d = q.popleft()
        if d == r:
            continue
        for u in adj[v]:
            if u not in vis:
                vis.add(u)
                q.append((u, d + 1))
    return vis

def _components(nodes, adj):
    seen = set()
    comps = []
    for v in nodes:
        if v in seen:
            continue
        c = []
        q = deque([v])
        seen.add(v)
        while q:
            x = q.popleft()
            c.append(x)
            for u in adj[x]:
                if u not in seen:
                    seen.add(u)
                    q.append(u)
        comps.append(c)
    return comps


# ---------- k-colorable connected graph ----------
def gen_k_colorable_graph(n, k, p, rng):
    assert 2 <= k <= len(PALETTE), "k must be between 2 and the palette size"
    assert n >= 1, "n must be >= 1"

    nodes = _labels(n)
    rng.shuffle(nodes)
    parts = [[] for _ in range(k)]
    for i, v in enumerate(nodes):
        parts[i % k].append(v)

    v2cidx = {}
    for ci, bucket in enumerate(parts):
        for v in bucket:
            v2cidx[v] = ci

    edges = set()
    for i in range(k):
        for j in range(i + 1, k):
            a, b = parts[i], parts[j]
            for u in a:
                for v in b:
                    if rng.random() < p:
                        e = (u, v) if u < v else (v, u)
                        edges.add(e)

    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)
    for v in nodes:
        adj[v]  # ensure key

    # ensure connectivity by stitching components with cross-class edges
    comps = _components(nodes, adj)
    if len(comps) > 1:
        for t in range(len(comps) - 1):
            A, B = comps[t], comps[t + 1]
            rng.shuffle(A)
            rng.shuffle(B)
            linked = False
            for u in A:
                if linked:
                    break
                for v in B:
                    if v2cidx[u] != v2cidx[v]:
                        e = (u, v) if u < v else (v, u)
                        if e not in edges:
                            edges.add(e)
                            adj[u].add(v)
                            adj[v].add(u)
                        linked = True
                        break

    return nodes, sorted([list(e) for e in edges]), v2cidx


# ---------- choose visibility radius ----------
def choose_radius(nodes, edges, s, target_frac=(0.06, 0.15)):
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    n = len(nodes)
    if n <= 10:
        return 1

    lo, hi = target_frac
    d = _avg_deg(n, edges) or 1.8
    d = max(1.5, min(d, 6.0))

    tgt = (lo + hi) * 0.5
    r = max(1, min(5, int(round(math.log(max(2.0, n * tgt), d)))))

    for _ in range(8):
        frac = len(_bfs_vis(adj, s, r)) / n
        if frac < lo and r < 5:
            r += 1
        elif frac > hi and r > 1:
            r -= 1
        else:
            break

    return max(1, min(5, r))


# ---------- build + save ----------
def build_level(n, k, p, pre_frac, start_mode):
    rng = random.Random()

    nodes, edges, v2cidx = gen_k_colorable_graph(n, k, p, rng)

    # adjacency for start-node choice
    adj = defaultdict(set)
    for u, v in edges:
        adj[u].add(v)
        adj[v].add(u)

    if start_mode == "random":
        s = rng.choice(nodes)
    else:
        s = max(nodes, key=lambda v: (len(adj[v]), v))

    r = choose_radius(nodes, edges, s)

    colors = PALETTE[:k]
    num_pre = max(0, int(pre_frac * n))
    pre_nodes = rng.sample(nodes, num_pre) if num_pre > 0 else []
    pre = {v: colors[v2cidx[v]] for v in pre_nodes}

    lvl = {
        "graph": {
            "nodes": nodes,
            "edges": edges
        },
        "pre_colored": pre,
        "colors": colors,
        "start_node": s,
        "visibility_radius": r
    }
    return lvl

def save_level(level, path):
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(level, f, indent=2)


# ---------- main ----------
if __name__ == "__main__":
    lvl = build_level(
        n=N_NODES,
        k=K_COLORS,
        p=EDGE_P,
        pre_frac=PRE_FRAC,
        start_mode=START_MODE
    )
    save_level(lvl, OUT_PATH)
