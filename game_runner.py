import json
from game_engine import GraphColoringGame
from student_template import CSP_AGENT 

class GameRunner:
    """
    The trusted "Referee" for the new assignment rules. It enforces the
    "Move-Then-Color" two-phase turn cycle.
    """
    def __init__(self, level_file, agent_class):
        self.game = GraphColoringGame(level_file)
        self.agent = agent_class(self.game.get_visible_state())
        self.max_steps = len(self.game.nodes) * 10 # Arbitrary large limit to prevent infinite loops.  

    def run_game(self):
        """
        Runs the new two-phase game loop.
        """
        print(f"Starting level. Agent at: {self.game.current_node}")

        for step in range(self.max_steps):
            print(f"\n--- Step {step + 1} ---")
            
            # --- PHASE 1: GET MOVE DECISION ---
            print(f"Agent is at '{self.game.current_node}'. Requesting next move...")
            visible_state = self.game.get_visible_state()
            try:
                move_action = self.agent.get_next_move(visible_state)
            except Exception as e:
                return self._fail_game(f"Agent crashed in get_next_move: {e}")
            
            is_valid, message = self._validate_move(move_action, visible_state)
            if not is_valid:
                return self._fail_game(f"Invalid move action: {message}")

            self.game.move_to(move_action['node'])
            print(f"Referee: Moved agent to '{self.game.current_node}'.")

            # --- PHASE 2: FORCE COLOR DECISION ---
            print(f"Agent is now at '{self.game.current_node}'. Requesting color...")
            visible_state_after_move = self.game.get_visible_state()
            try:
                color_action = self.agent.get_color_for_node(self.game.current_node, visible_state_after_move)
            except Exception as e:
                return self._fail_game(f"Agent crashed in get_color_for_node: {e}")

            is_valid, message = self._validate_color(color_action, visible_state_after_move)
            if not is_valid:
                return self._fail_game(f"Invalid color action: {message}")

            self.game.assign_color(color_action['node'], color_action['color'])
            print(f"Referee: Assigned color '{color_action['color']}' to node '{color_action['node']}'.")

            if self.game.is_fully_and_correctly_colored():
                print("\n--- Puzzle Solved! ---")
                break
        
        if not self.game.is_fully_and_correctly_colored():
            print("\n--- Max steps reached or puzzle incorrect. Game Over. ---")

        return self.game.get_final_summary()

    def _fail_game(self, error_message):
        """Handles a disqualification and returns a zero-score summary."""
        print(f"\n--- AGENT DISQUALIFIED ---")
        print(error_message)
        summary = self.game.get_final_summary()
        summary['score'] = 0
        summary['is_correct'] = False
        summary['error'] = error_message
        return summary

    def _validate_move(self, action, state):
        """Validates a 'move' action."""
        if not isinstance(action, dict) or action.get('action') != 'move':
            return False, "Action must be a dictionary {'action': 'move', 'node': 'NODE_ID'}."
        node = action.get('node')
        # --- THE FIX: Access the nested 'nodes' list ---
        if node not in state['visible_graph']['nodes']:
            return False, f"Cannot move to node '{node}' as it is not currently visible."
        return True, "OK"

    def _validate_color(self, action, state):
        """Validates a 'color' action."""
        if not isinstance(action, dict) or action.get('action') != 'color':
            return False, "Action must be a dictionary {'action': 'color', 'node': 'NODE_ID', 'color': 'COLOR'}."
        node = action.get('node')
        color = action.get('color')
        if node != state['current_node']:
            return False, f"Can only color the current node '{state['current_node']}', not '{node}'."
        if color not in state['available_colors']:
            return False, f"Color '{color}' is not valid for this level."
        return True, "OK"

if __name__ == "__main__":
    level_file = "level5.json"
    level_data = {
        "graph": {
            "nodes": ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "C3"],
            "edges": [["A1","A2"], ["A2","A3"], ["B1","B2"], ["B2","B3"], ["C1","C2"], ["C2","C3"],
                    ["A1","B1"], ["A2","B2"], ["A3","B3"], ["B1","C1"], ["B2","C2"], ["B3","C3"],
                    ["A1","B2"], ["A2","B3"], ["B1","C2"], ["B2","C3"]]
        },
        "pre_colored": {"A1":"Blue", "C3":"Red"},
        "colors": ["Red", "Green", "Blue"],
        "start_node": "B2",
        "visibility_radius": 1
        }

    with open(level_file, 'w') as f: json.dump(level_data, f)

    runner = GameRunner(level_file, CSP_AGENT)
    final_summary = runner.run_game()

    print("\n" + "="*20 + " FINAL SUMMARY " + "="*20)
    print(json.dumps(final_summary, indent=2))

