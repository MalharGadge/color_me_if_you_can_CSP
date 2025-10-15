import json
from collections import deque, defaultdict
import math

class GraphColoringGame:
    def __init__(self, level_file):
        with open(level_file) as f:
            data = json.load(f)
        
        self.nodes = data["graph"]["nodes"]
        self.edges = data["graph"]["edges"]
        self.colors = data["colors"]
        self.visibility_radius = data["visibility_radius"]
        self.start_node = data["start_node"]
        self.pre_colored = data.get("pre_colored", {})
        
        self.adj = defaultdict(list)
        for u, v in self.edges:
            self.adj[u].append(v)
            self.adj[v].append(u)
            
        self.node_colors = {n: self.pre_colored.get(n, None) for n in self.nodes}
        self.reassignments = 0
        self.moves = 0
        self.current_node = self.start_node

    def get_visible_state(self):
        """
        Returns the limited, partially observable state for the agent.
        """
        queue = deque([(self.current_node, 0)])
        visited_bfs = {self.current_node}
        visible_nodes = {self.current_node}
        visible_edges = set()

        while queue:
            node, distance = queue.popleft()
            if distance >= self.visibility_radius:
                continue
            
            for neighbor in self.adj[node]:
                visible_edges.add(tuple(sorted((node, neighbor))))
                if neighbor not in visited_bfs:
                    visited_bfs.add(neighbor)
                    visible_nodes.add(neighbor)
                    queue.append((neighbor, distance + 1))
        
        return {
            "current_node": self.current_node,
            "available_colors": self.colors,
            "visible_graph": {
                "nodes": list(visible_nodes),
                "edges": [list(e) for e in visible_edges if e[0] in visible_nodes and e[1] in visible_nodes],
            },
            "node_colors": {n: self.node_colors[n] for n in visible_nodes}
        }

    def move_to(self, node):
        """Updates the agent's current position and tracks move counts."""
        if node != self.current_node:
            self.moves += 1
        self.current_node = node
        return f"Moved to {node}."

    def assign_color(self, node, color):
        """Assigns a color and tracks reassignments."""
        if node in self.pre_colored:
            return "Cannot re-color a pre-colored node."
            
        if self.node_colors.get(node) is not None and self.node_colors.get(node) != color:
            self.reassignments += 1
        
        self.node_colors[node] = color
        return f"Colored {node} with {color}."

    def is_fully_and_correctly_colored(self):
        """Checks if the entire graph is solved."""
        if not all(self.node_colors.get(n) is not None for n in self.nodes):
            return False
        
        for u, v in self.edges:
            if self.node_colors.get(u) == self.node_colors.get(v):
                return False
        
        return True

    def get_final_summary(self):
        """
        Calculates the final score, assigning negative infinity to any failed solution.
        """
        is_correct = self.is_fully_and_correctly_colored()
        
        if is_correct:
            total_penalties = self.reassignments + self.moves
            final_score = 100 - total_penalties
            
            if self.reassignments == 0:
                final_score += 10
            
            final_score = round(final_score)
        else:
            final_score = -math.inf

        print(f"Final Score: {final_score} (Correct: {is_correct}, Moves: {self.moves}, Reassignments: {self.reassignments})")
        return {
            "node_colors": self.node_colors,
            "moves": self.moves,
            "reassignments": self.reassignments,
            "score": final_score,
            "is_correct": is_correct
        }

