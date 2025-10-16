# [YourRollNumber].py
# CSP agent with MRV, LCV, Forward Checking, AC-3,
# cycle-robust navigation (tabu window, edge-tabu, stuck detector),
# Nearest-Uncolored Routing (goal persistence + shortest-path first-hop),
# and a bounded multi-variable backtracking plan over the current visible subgraph.
#
# No prints/logging. Single file. Short, readable names.

import random
from collections import deque, defaultdict


class CSP_AGENT:
    def __init__(self, initial_state):
        self.colors = []
        self.r = 1

        seed = 0
        self.k_assign = 2
        self.k_explore = 2
        self.decay = 0.15
        self.tabu_len = 4
        self.stuck_limit = 2

        # backtracking planner limits (small to keep steps fast)
        self.bt_max_vars = 8          # plan on at most this many visible vars
        self.bt_max_expansions = 2000 # node expansions cap for recursion

        if isinstance(initial_state, dict):
            self.colors = list(initial_state.get('available_colors', []))
            self.r = int(initial_state.get('visibility_radius', 1))
            seed = int(initial_state.get('agent_seed', seed))
            self.k_assign = int(initial_state.get('agent_topk_assign', self.k_assign))
            self.k_explore = int(initial_state.get('agent_topk_explore', self.k_explore))
            self.decay = float(initial_state.get('agent_decay', self.decay))
            self.bt_max_vars = int(initial_state.get('agent_bt_max_vars', self.bt_max_vars))
            self.bt_max_expansions = int(initial_state.get('agent_bt_max_expansions', self.bt_max_expansions))

        self.rng = random.Random(seed)
        self.colors.sort()

        self.pos = None
        self.prev = None
        self.recent = deque(maxlen=self.tabu_len)
        self.edge_recent = deque(maxlen=self.tabu_len)

        self.adj = defaultdict(set)   # known graph (only edges we've seen)
        self.dom = {}                 # domains of seen nodes
        self.asg = {}                 # committed colors
        self.seen = set()             # ever-seen nodes
        self.vis = set()              # currently visible nodes
        self.vcount = defaultdict(int)

        self.since_color = 0          # moves since last new color commit
        self.last_vis = set()         # previous visible set
        self.since_reveal = 0         # moves since last new node was revealed

        # -------- NUR state --------
        self.goal = None              # current routing target = nearest seen-uncolored

    # ---------------- engine entrypoints ----------------

    def get_next_move(self, visible_state):
        self._sync(visible_state)
        self._ac3_pre()
        self._update_goal()  # refresh nearest-uncolored goal if needed

        force = (self.since_color >= self.stuck_limit or self.since_reveal >= self.stuck_limit)

        # If current node is a candidate to color now, stay (must-move rule).
        if self.pos in self.vis and self.pos not in self.asg and len(self.dom.get(self.pos, set())) > 0:
            return {'action': 'move', 'node': self.pos}

        # If any visible uncolored node has empty domain, explore (reveal more) first.
        if any(v not in self.asg and len(self.dom.get(v, set())) == 0 for v in self.vis):
            pool = [n for n in self.vis if n != self.pos]
            hop = self._route_hop(self.goal)
            if hop and hop in pool:
                return {'action': 'move', 'node': hop}
            nxt = self._choose_explore_node(pool, force=force) or self.pos
            return {'action': 'move', 'node': self._safe_visible(nxt)}

        # Assign next: unit-first, then general MRV among visible-uncolored.
        uncol = [v for v in self.vis if v not in self.asg and len(self.dom.get(v, set())) > 0]
        if uncol:
            unit = [v for v in uncol if len(self.dom[v]) == 1]
            scope = unit if unit else uncol

            if self.pos in scope:
                self.goal = self.pos
                return {'action': 'move', 'node': self.pos}

            cand = self._mrv_on(scope, force=force)
            tgt = self._choose_assign_node(cand, force=force)

            if tgt is None or tgt == self.prev:
                non_prev = [n for n in scope if n != self.prev]
                if non_prev:
                    tgt = self._choose_assign_node(non_prev, force=True) or non_prev[0]

            if tgt is None:
                others = [n for n in self.vis if n != self.pos]
                hop = self._route_hop(self.goal)
                if hop and hop in others:
                    tgt = hop
                else:
                    tgt = self._choose_explore_node(others, force=True) or (others[0] if others else self.pos)

            if tgt in scope:
                self.goal = tgt
            return {'action': 'move', 'node': self._safe_visible(tgt)}

        # All visible are colored → exploration.
        others = [n for n in self.vis if n != self.pos]
        if others:
            hop = self._route_hop(self.goal)
            if hop and hop in others:
                return {'action': 'move', 'node': hop}
            nxt = self._choose_explore_node(others, force=force) or self.pos
            return {'action': 'move', 'node': self._safe_visible(nxt)}

        return {'action': 'move', 'node': self.pos}

    def get_color_for_node(self, node_to_color, visible_state):
        self._sync(visible_state)

        v = node_to_color
        shown = visible_state.get('node_colors', {}) or {}
        cur = shown.get(v)

        # Mirror engine color if already set (no reassignment).
        if cur is not None:
            was_new = v not in self.asg
            self.asg[v] = cur
            self.dom[v] = {cur}
            if was_new:
                self.since_color = 0
                if self.goal == v:
                    self.goal = None
            return {'action': 'color', 'node': v, 'color': cur}

        self._ac3_pre()

        # --- NEW: bounded multi-variable backtracking plan on current visible subgraph ---
        plan = self._plan_visible_coloring(priority=v)
        if plan and v in plan:
            c = plan[v]
            if self._safe_assign(v, c):
                was_new = v not in self.asg
                self.asg[v] = c
                self.dom[v] = {c}
                if was_new:
                    self.since_color = 0
                    if self.goal == v:
                        self.goal = None
                return {'action': 'color', 'node': v, 'color': c}
        # -------------------------------------------------------------------------------

        # If domain empty (rare), pick visible-legal color to keep progress.
        if v not in self.dom or len(self.dom[v]) == 0:
            c = self._first_visible_legal(v, visible_state)
            if c is None:
                c = (self.colors[0] if self.colors else None)
            was_new = v not in self.asg
            self.asg[v] = c
            self.dom[v] = {c} if c is not None else set()
            if was_new:
                self.since_color = 0
                if self.goal == v:
                    self.goal = None
            return {'action': 'color', 'node': v, 'color': c}

        # Try LCV + incremental AC-3 (safe commit) as before.
        for c in self._lcv(v):
            if self._safe_assign(v, c):
                was_new = v not in self.asg
                self.asg[v] = c
                self.dom[v] = {c}
                if was_new:
                    self.since_color = 0
                    if self.goal == v:
                        self.goal = None
                return {'action': 'color', 'node': v, 'color': c}

        # Fallback: any color legal w.r.t. currently visible neighbors.
        c = self._first_visible_legal(v, visible_state)
        if c is None:
            pool = self.dom.get(v, set()) or set(self.colors)
            c = (sorted(pool)[0] if pool else None)
        was_new = v not in self.asg
        self.asg[v] = c
        self.dom[v] = {c} if c is not None else set()
        if was_new:
            self.since_color = 0
            if self.goal == v:
                self.goal = None
        return {'action': 'color', 'node': v, 'color': c}

    # ---------------- sync ----------------

    def _sync(self, vs):
        if not self.colors:
            self.colors = list(vs.get('available_colors', []))
            self.colors.sort()
        self.r = int(vs.get('visibility_radius', self.r))

        new_pos = vs.get('current_node', self.pos)
        if new_pos is not None:
            if self.pos is not None and new_pos != self.pos:
                self.prev = self.pos
                self.edge_recent.append((self.prev, new_pos))
                self.recent.append(self.pos)
                self.since_color += 1
                self.since_reveal += 1
            self.pos = new_pos
            self.vcount[self.pos] += 1

        g = vs.get('visible_graph', {}) or {}
        nodes = list(g.get('nodes', []))
        edges = [tuple(e) for e in g.get('edges', [])]

        self.vis = set(nodes)
        newly = [v for v in self.vis if v not in self.seen]
        if newly:
            self.since_reveal = 0
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

        self.last_vis = set(self.vis)

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

    def _mrv_on(self, scope, force=False):
        m = min(len(self.dom.get(v, set())) for v in scope)
        cand = [v for v in scope if len(self.dom.get(v, set())) == m]
        if len(cand) <= 1:
            return cand
        key = lambda v: (self._cycle_penalty(v), -self._gain_uncol(v), self.vcount[v], v)
        if force or len(self.seen) <= 50:
            return sorted(cand, key=key)
        return self._pick_fast_decay(cand, key_fn=key, top_k=self.k_assign)

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

    # ---------------- NUR: goal update + routing ----------------

    def _update_goal(self):
        if self.goal is not None and self.goal in self.asg:
            self.goal = None
        if self.goal is None or self.goal not in self.seen or self.goal in self.asg:
            g = self._nearest_uncolored()
            self.goal = g if g is not None else None

    def _nearest_uncolored(self):
        if self.pos is None:
            return None
        tgt = [v for v in self.seen if v not in self.asg and v != self.pos]
        if not tgt:
            return None
        q = deque([self.pos])
        par = {self.pos: None}
        while q:
            x = q.popleft()
            if x in tgt:
                return x
            for u in self.adj[x]:
                if u in self.seen and u not in par:
                    par[u] = x
                    q.append(u)
        return None

    def _route_hop(self, goal):
        if self.pos is None or goal is None or goal == self.pos:
            return None
        q = deque([self.pos])
        par = {self.pos: None}
        while q:
            x = q.popleft()
            if x == goal:
                break
            for u in self.adj[x]:
                if u in self.seen and u not in par:
                    par[u] = x
                    q.append(u)
        if goal not in par:
            return None
        y = goal
        while par[y] is not None and par[y] != self.pos:
            y = par[y]
        hop = y
        if hop not in self.vis:
            return None
        if (self.pos, hop) in self.edge_recent or (hop, self.pos) in self.edge_recent:
            return None
        return hop

    # ---------------- movement heuristics ----------------

    def _choose_assign_node(self, cand, force=False):
        if not cand:
            return None
        pool = self._avoid_recent(cand)
        key = lambda v: (self._cycle_penalty(v), -self._gain_uncol(v), -self._gain_unseen(v), self.vcount[v], v)
        if force or len(self.seen) <= 50:
            return sorted(pool, key=key)[0] if pool else None
        return self._pick_fast_decay(pool, key_fn=key, top_k=self.k_assign)

    def _choose_explore_node(self, cand, force=False):
        if not cand:
            return None
        pos = [v for v in cand if self._gain_unseen(v) > 0]
        scope = pos if pos else list(cand)
        scope = self._avoid_recent(scope)
        key = lambda v: (self._cycle_penalty(v), -self._gain_unseen(v), -self._gain_uncol(v), self.vcount[v], v)
        if force or len(self.seen) <= 50:
            return sorted(scope, key=key)[0] if scope else None
        return self._pick_fast_decay(scope, key_fn=key, top_k=self.k_explore)

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
        return (e, len(nbrs))

    def _cycle_penalty(self, v):
        p = 0
        if v == self.prev:
            p += 1000
        if v in self.recent:
            p += 200
        if (self.pos, v) in self.edge_recent or (v, self.pos) in self.edge_recent:
            p += 500
        p += self.vcount[v] * 5
        return p

    def _safe_visible(self, n):
        return n if n in self.vis else self.pos

    def _avoid_recent(self, cand):
        if not cand:
            return []
        vis_pool = [v for v in cand if v in self.vis]
        if not vis_pool:
            return []
        recent = set(self.recent)
        non_recent = [
            v for v in vis_pool
            if v not in recent and v != self.prev
            and (self.pos, v) not in self.edge_recent and (v, self.pos) not in self.edge_recent
        ]
        if non_recent:
            return non_recent
        non_prev = [v for v in vis_pool if v != self.prev]
        if non_prev:
            return sorted(non_prev, key=lambda v: self._cycle_penalty(v))
        return sorted(vis_pool, key=lambda v: self._cycle_penalty(v))

    # ---------------- fast-decay randomized picker ----------------

    def _pick_fast_decay(self, cand, key_fn, top_k=1):
        lst = [v for v in cand if v in self.vis]
        if not lst:
            return None
        items = [(key_fn(v), v) for v in lst]
        items.sort(key=lambda t: t[0])

        keys = []
        buckets = defaultdict(list)
        for k, v in items:
            buckets[k].append(v)
            if not keys or k != keys[-1]:
                keys.append(k)

        eff_topk = 1 if len(self.seen) <= 50 else max(1, int(top_k))
        use_keys = keys[:eff_topk]

        pool, wts = [], []
        for r, k in enumerate(use_keys):
            w = (0.05 if len(self.seen) <= 50 else self.decay) ** r
            for v in buckets[k]:
                pool.append(v)
                wts.append(w)

        s = sum(wts)
        if s <= 0 or len(pool) == 1:
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

    # ---------------- bounded backtracking planner (NEW) ----------------

    def _plan_visible_coloring(self, priority=None):
        """
        Try to find a consistent assignment for a small subset of the visible frontier,
        prioritizing `priority` (usually the node the referee asked us to color).
        Returns a dict {node: color} if a plan is found; otherwise None.
        """
        # candidate variables = visible & uncolored with non-empty domains
        vars_all = [v for v in self.vis if v not in self.asg and len(self.dom.get(v, set())) > 0]
        if not vars_all:
            return {}

        # order: put priority first (if present), then MRV-based order
        order_rest = self._order_vars([x for x in vars_all if x != priority])
        order = [priority] + order_rest if (priority in vars_all) else order_rest

        # size cap to keep search tiny and fast
        order = order[:self.bt_max_vars]

        dom0 = {x: set(self.dom[x]) for x in self.dom}  # copy
        fixed = {x: self.asg[x] for x in self.asg if x in self.vis}  # visible fixed colors

        # AC-3 pre on the working copy (visible arcs only)
        q = []
        for x in self.vis:
            for y in self.adj[x]:
                if y in self.vis and x != y:
                    q.append((x, y))
        if not self._ac3(q, dom0):
            return None
        if any(len(dom0.get(x, set())) == 0 for x in order):
            return None

        # depth-first backtracking with FC + incremental AC-3
        plan, ok = self._bt_solve(order, 0, dom0, fixed, 0, self.bt_max_expansions)
        return plan if ok else None

    def _order_vars(self, vars_list):
        """MRV, then more uncolored neighbors, then smaller cycle key, then id."""
        if not vars_list:
            return []
        return sorted(vars_list,
                      key=lambda v: (len(self.dom.get(v, set())),
                                     -self._gain_uncol(v),
                                     self._cyc_key(v),
                                     v))

    def _bt_solve(self, order, idx, dom, fixed, exp, cap):
        """
        Backtracking search over 'order' using LCV + incremental AC-3.
        - dom: working domains (mutable copy)
        - fixed: visible nodes already colored (we never change them)
        - exp/cap: expansion counter/limit
        Returns (plan_dict, True) or ({}, False).
        """
        if idx == len(order):
            return ({}, True)

        if exp >= cap:
            return ({}, False)

        v = order[idx]
        vals = sorted(dom.get(v, set()))
        if not vals:
            return ({}, False)

        # LCV scoring (fewer prunes to neighbors)
        nbrs = [u for u in self.adj[v] if u in self.vis]
        sc = {}
        for c in vals:
            s = 0
            for u in nbrs:
                if c in dom.get(u, set()):
                    s += 1
            sc[c] = s
        vals.sort(key=lambda c: (sc[c], c))

        for c in vals:
            exp += 1
            # clone dom, assign v=c, then incremental AC-3 on arcs touching v
            dom2 = {x: set(dom[x]) for x in dom}
            dom2[v] = {c}
            q = [(u, v) for u in self.adj[v] if u in self.vis]
            if not self._ac3(q, dom2):
                continue
            if any(len(dom2.get(x, set())) == 0 for x in self.vis):
                continue

            bad = False
            for x, col in fixed.items():
                d = dom2.get(x, set())
                if not d or (len(d) == 1 and next(iter(d)) != col):
                    bad = True
                    break
            if bad:
                continue

            subplan, ok = self._bt_solve(order, idx + 1, dom2, fixed, exp, cap)
            if ok:
                subplan[v] = c
                return (subplan, True)

        return ({}, False)
