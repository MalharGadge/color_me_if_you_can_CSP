# [YourRollNumber].py
# CSP agent with MRV, LCV, Forward Checking, AC-3,
# and guarded fast-decay randomized tie-breaks.
# Single-file. No prints/logging. No external libs.

import random
from collections import deque, defaultdict


class CSP_AGENT:
    def __init__(self, initial_state):
        self.colors = []
        self.r = 1

        seed = 0
        self.k_assign = 2       # consider top-k keys for assignment tie-break
        self.k_explore = 2      # consider top-k keys for exploration tie-break
        self.decay = 0.15       # fast-decay prob across ranks (0<decay<1, smaller = faster drop)

        if isinstance(initial_state, dict):
            self.colors = list(initial_state.get('available_colors', []))
            self.r = int(initial_state.get('visibility_radius', 1))
            seed = int(initial_state.get('agent_seed', seed))
            self.k_assign = int(initial_state.get('agent_topk_assign', self.k_assign))
            self.k_explore = int(initial_state.get('agent_topk_explore', self.k_explore))
            self.decay = float(initial_state.get('agent_decay', self.decay))

        self.rng = random.Random(seed)
        self.colors.sort()

        self.pos = None
        self.adj = defaultdict(set)   # node -> set(neighbors)
        self.dom = {}                 # node -> set(colors)
        self.asg = {}                 # node -> color
        self.seen = set()             # ever seen
        self.vis = set()              # currently visible

    # ---------------- engine entrypoints ----------------

    def get_next_move(self, visible_state):
        self._sync(visible_state)
        self._ac3_pre()

        # 1) If any visible uncolored node has empty domain, reveal more before committing.
        if any(v not in self.asg and len(self.dom.get(v, set())) == 0 for v in self.vis):
            pool = [n for n in self.vis if n != self.pos]
            nxt = self._choose_explore_node(pool) or self.pos
            return {'action': 'move', 'node': self._safe_visible(nxt)}

        # 2) Assign next: choose visible, uncolored nodes with non-empty domain.
        uncol = [v for v in self.vis if v not in self.asg and len(self.dom.get(v, set())) > 0]
        if uncol:
            cand = self._mrv_on(uncol)
            tgt = self._choose_assign_node(cand)

            # Guard: if tie-break returns None or the current node (shouldn't), try best alternative.
            if tgt is None or tgt == self.pos:
                alt = [n for n in uncol if n != self.pos]
                if alt:
                    tgt = self._choose_assign_node(alt) or alt[0]
                else:
                    tgt = self.pos

            return {'action': 'move', 'node': self._safe_visible(tgt)}

        # 3) All visible are colored → explore to expand frontier.
        others = [n for n in self.vis if n != self.pos]
        if others:
            nxt = self._choose_explore_node(others) or self.pos
            return {'action': 'move', 'node': self._safe_visible(nxt)}

        return {'action': 'move', 'node': self.pos}

    def get_color_for_node(self, node_to_color, visible_state):
        self._sync(visible_state)

        v = node_to_color
        shown = visible_state.get('node_colors', {}) or {}
        cur = shown.get(v)

        # If engine already set a color, mirror it (avoid recolor).
        if cur is not None:
            self.asg[v] = cur
            self.dom[v] = {cur}
            return {'action': 'color', 'node': v, 'color': cur}

        # Enforce arc consistency on current visible subgraph.
        self._ac3_pre()

        # If domain is empty (rare under AC-3), pick a visible-legal color to keep progress.
        if v not in self.dom or len(self.dom[v]) == 0:
            c = self._first_visible_legal(v, visible_state)
            if c is None:
                c = (self.colors[0] if self.colors else None)
            self.asg[v] = c
            self.dom[v] = {c} if c is not None else set()
            return {'action': 'color', 'node': v, 'color': c}

        # Try LCV with incremental AC-3 for safe commit (avoid future reassignments).
        for c in self._lcv(v):
            if self._safe_assign(v, c):
                self.asg[v] = c
                self.dom[v] = {c}
                return {'action': 'color', 'node': v, 'color': c}

        # Fallback: a color legal wrt currently visible neighbors.
        c = self._first_visible_legal(v, visible_state)
        if c is None:
            pool = self.dom.get(v, set()) or set(self.colors)
            c = (sorted(pool)[0] if pool else None)
        self.asg[v] = c
        self.dom[v] = {c} if c is not None else set()
        return {'action': 'color', 'node': v, 'color': c}

    # ---------------- sync ----------------

    def _sync(self, vs):
        if not self.colors:
            self.colors = list(vs.get('available_colors', []))
            self.colors.sort()
        self.r = int(vs.get('visibility_radius', self.r))

        self.pos = vs.get('current_node', self.pos)

        g = vs.get('visible_graph', {}) or {}
        nodes = list(g.get('nodes', []))
        edges = [tuple(e) for e in g.get('edges', [])]

        self.vis = set(nodes)
        self.seen.update(self.vis)

        for u, w in edges:
            self.adj[u].add(w)
            self.adj[w].add(u)

        for v in self.vis:
            if v not in self.dom:
                self.dom[v] = set(self.colors)

        shown = vs.get('node_colors', {}) or {}
        for v, c in shown.items():
            self.seen.add(v)
            self.adj[v]
            if v not in self.dom:
                self.dom[v] = set(self.colors)
            if c is not None:
                self.dom[v] = {c}
                self.asg[v] = c

        for v in list(self.dom.keys()):
            if v not in self.seen:
                del self.dom[v]

    # ---------------- CSP core ----------------

    def _ac3_pre(self):
        q = []
        for x in self.vis:
            for y in self.adj[x]:
                if y in self.vis and x != y:
                    q.append((x, y))
        self._ac3(q, self.dom)

    def _ac3(self, q, dom):
        Q = deque(q)
        while Q:
            x, y = Q.popleft()
            if self._revise(x, y, dom):
                if len(dom[x]) == 0:
                    return False
                for z in self.adj[x]:
                    if z in self.vis and z != y:
                        Q.append((z, x))
        return True

    def _revise(self, x, y, dom):
        if x not in dom or y not in dom or len(dom[y]) == 0:
            return False
        if len(dom[y]) > 1:
            return False

        a_y = next(iter(dom[y]))
        if a_y in dom[x]:
            dom[x] = dom[x] - {a_y}
            return True
        return False

    def _mrv_on(self, scope):
        # MRV primary: smaller domain first
        m = min(len(self.dom.get(v, set())) for v in scope)
        cand = [v for v in scope if len(self.dom.get(v, set())) == m]
        if len(cand) <= 1:
            return cand

        # Secondary: more uncolored neighbors (maximize); Tertiary: tree-ness key
        return self._pick_fast_decay(
            cand,
            key_fn=lambda v: (-self._gain_uncol(v), self._cyc_key(v)),
            top_k=self.k_assign
        )

    def _lcv(self, v):
        vals = sorted(self.dom.get(v, set()))
        if len(vals) <= 1:
            return vals

        nbrs = [u for u in self.adj[v] if u in self.vis]
        sc = {}
        for c in vals:
            s = 0
            for u in nbrs:
                if c in self.dom.get(u, set()):
                    s += 1
            sc[c] = s

        vals.sort(key=lambda c: (sc[c], c))
        return vals

    def _safe_assign(self, v, c):
        dom2 = {x: set(self.dom[x]) for x in self.dom}
        dom2[v] = {c}

        q = [(u, v) for u in self.adj[v] if u in self.vis]
        ok = self._ac3(q, dom2)
        if not ok:
            return False
        if any(len(dom2[x]) == 0 for x in self.vis):
            return False

        self.dom = dom2
        return True

    # ---------------- movement heuristics ----------------

    def _choose_assign_node(self, cand):
        if not cand:
            return None
        # Prefer higher uncolored gain, then tree-ness
        return self._pick_fast_decay(
            cand,
            key_fn=lambda v: (-self._gain_uncol(v), self._cyc_key(v)),
            top_k=self.k_assign
        )

    def _choose_explore_node(self, cand):
        if not cand:
            return None
        # Explore: uncolored gain, then unseen gain, then tree-ness
        return self._pick_fast_decay(
            cand,
            key_fn=lambda v: (-self._gain_uncol(v), -self._gain_unseen(v), self._cyc_key(v)),
            top_k=self.k_explore
        )

    # ---------------- frontier & structure helpers ----------------

    def _gain_uncol(self, v):
        return sum(1 for u in self.adj[v] if u in self.vis and u not in self.asg)

    def _gain_unseen(self, v):
        return sum(1 for u in self.adj[v] if u not in self.seen)

    def _cyc_key(self, v):
        nbrs = [u for u in self.adj[v] if u in self.vis]
        e = 0
        for i in range(len(nbrs)):
            a = nbrs[i]
            for j in range(i + 1, len(nbrs)):
                b = nbrs[j]
                if b in self.adj[a]:
                    e += 1
        return (e, len(nbrs), v)

    def _safe_visible(self, n):
        return n if n in self.vis else self.pos

    # ---------------- fast-decay randomized picker ----------------

    def _pick_fast_decay(self, cand, key_fn, top_k=1):
        """
        Sort by key_fn (ascending), then select from top_k DISTINCT key-buckets.
        Within ranks, use fast-decay weights: w(rank=r) = decay^r with 0<decay<1 (fast drop).
        Always returns a node from self.vis.
        """
        lst = [v for v in cand if v in self.vis]
        if not lst:
            return None

        items = [(key_fn(v), v) for v in lst]
        items.sort(key=lambda t: t[0])

        # Build buckets by distinct keys in order
        keys = []
        buckets = defaultdict(list)
        for k, v in items:
            buckets[k].append(v)
            if not keys or k != keys[-1]:
                keys.append(k)

        # Limit to top_k key buckets
        k_lim = max(1, top_k)
        use_keys = keys[:k_lim]

        # Build selection pool with rank-based weights (rank = index in use_keys)
        pool = []
        wts = []
        for r, k in enumerate(use_keys):
            w = self.decay ** r  # fast decay as rank increases
            for v in buckets[k]:
                pool.append(v)
                wts.append(w)

        # Normalize weights and sample
        s = sum(wts)
        if s <= 0:
            return pool[0]
        x = self.rng.random() * s
        acc = 0.0
        for v, w in zip(pool, wts):
            acc += w
            if x <= acc:
                return v
        return pool[-1]

    # ---------------- fallback visible-legal ----------------

    def _first_visible_legal(self, v, vs):
        used = set()
        g = vs.get('visible_graph', {}) or {}
        edges = [tuple(e) for e in g.get('edges', [])]
        shown = vs.get('node_colors', {}) or {}

        for a, b in edges:
            if a == v and b in shown and shown[b] is not None:
                used.add(shown[b])
            elif b == v and a in shown and shown[a] is not None:
                used.add(shown[a])

        pool = sorted(self.dom.get(v, set()) or self.colors)
        for c in pool:
            if c not in used:
                return c
        return None
