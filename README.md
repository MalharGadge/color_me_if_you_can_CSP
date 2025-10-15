# Colour Me If You Can: CSP Tournament Assignment

A competitive CSP programming assignment where students implement intelligent agents to solve graph coloring problems under partial observability. This assignment combines Constraint Satisfaction Problems (CSP) with the challenge of limited visibility of radius 1 in a tournament setting.

## Assignment Objective

Your task is to develop an intelligent agent to solve the **Graph Coloring Problem** on a graph where only a local portion is visible at any given time. Your agent must utilize CSP reasoning, heuristics, and backtracking search to find a complete and valid coloring for the graph with **minimal reassignments (backtracks)**.

**Primary Goal**: Build a robust agent that can effectively handle partial information to complete the coloring task while aiming for the lowest number of reassignments.

⚠️ **Important Notes:**
- We highly encourage you to go through the entire codebase for better understanding
- **Do not modify** the underlying CSP engine code - evaluation uses the original copy
- Your **only contribution** is the AI agent implementation the `student_template.py` file

## Project Structure


- `student_template.py`: Template file to get you started (**DO FOLLOW NAMING CONVENTIONS AS MENTIONED.**)
- `game_engine.py`: Core CSP engine and game logic (**DO NOT MODIFY**)
- `game_runner.py`: Game referee and validation system (**DO NOT MODIFY**)
- `level1.json`: Sample level configuration for testing your agent

### Key Components

#### Your Implementation (`[YourRollNumber].py`)
This is the **only file you should submit**. Copy the template and implement the `Your_ROLL_NUMBER` class:
- Must implement a **backtracking CSP solver**
- All logic must be contained in a single Python file

#### CSP Engine (`game_engine.py`)
- Constraint validation and propagation
- Graph coloring constraint checking
- Visibility radius management
- Move and color action processing

#### Game Runner (`game_runner.py`)
- Tournament game flow control
- Agent action validation and scoring
- Reassignment/backtrack detection
- Final scoring calculation

#### Level Configuration (`level1.json`)
Sample level with:
- 5-node graph (A, B, C, D, E)
- 3 available colors (Red, Green, Blue)
- Visibility radius of 1 (can only see immediate neighbors)
- Starting position at node A

## Game Rules and Constraints

### 🎯 Goal State
**All nodes are assigned a valid color**, meaning no two adjacent nodes share the same color.

### 👁️ Partial Observability  
The agent sees only the subgraph of nodes within a defined **visibility radius of 1** of the current node. Unseen nodes are unknown until explored.

### 🚶 Agent Movement
The agent can:
- Move to any **uncolored node** within the visible subgraph
- Choose to **stay at the current node**
- Movement must be strategic for gathering information about the graph structure

### 🎨 Color Assignment
Each node must be assigned a color from a **fixed, allowed set** (e.g., Red, Green, Blue) as defined in the level JSON file.

## Constraints

### ⚠️ Coloring Constraint
**Adjacent nodes cannot share the same color** (standard graph coloring constraint).

### 🔄 Algorithm Constraint  
Your agent **must implement a backtracking CSP solver**. You cannot use:
- Local Search algorithms
- Reinforcement Learning approaches
- External graph coloring libraries

### 📁 Code Structure Constraint
- Your agent's logic must be contained within a **single Python file** named `[YourRollNumber].py`
- **Do not modify** other helper files like `game_engine.py`
- Submit only your agent file

## Scoring System

The tournament uses a comprehensive scoring system designed to reward efficient CSP solving:

### 🏆 Points Breakdown

| Action | Points | Description |
|--------|--------|-------------|
| **Correct Coloring** | **+100** | Base points awarded only if the entire graph is fully and correctly colored |
| **Move Penalty** | **-1** | Deducted for every move action (cost of exploration) |
| **Reassignment/Backtrack Penalty** | **-1** | Deducted each time you change a node's color (trial and error cost) |
| **Perfection Bonus** | **+10** | Awarded only for correct coloring with **zero reassignments** |
| **Failure Score** | **-∞** | If graph is not fully and correctly colored at game end |

### 📊 Final Score Calculation

```
Final Score = Base Score - Total Moves - Total Reassignments + Bonuses
```

**Examples:**
- Perfect solution in 8 moves, 0 reassignments: `100 - 8 - 0 + 10 = 102 points`
- Good solution in 12 moves, 3 reassignments: `100 - 12 - 3 + 0 = 85 points`
- Failed solution: `-∞ points`

**Notes:**
- Scores can be negative if penalties exceed base score and bonuses
- Maximum possible score is **110 points** (minus minimum required moves)
- **Reassignments are heavily penalized** - plan your coloring strategy carefully!

## Getting Started

### Setup
1. **Copy** `student_template.py` to `[YourRollNumber].py`
2. **Implement** your CSP agent in the new file
3. **Test** your implementation:
   ```bash
   python game_runner.py
   ```



## Level 1 Example

The provided `level1.json` contains:
```json
{
  "graph": {
    "nodes": ["A", "B", "C", "D", "E"],
    "edges": [["A","B"], ["B","C"], ["C","D"], ["D","E"], ["E","A"], ["A","C"]]
  },
  "pre_colored": {},
  "colors": ["Red", "Green", "Blue"],
  "start_node": "A",
  "visibility_radius": 1
}
```

**Graph Structure**: 5-node cycle with diagonal edge (A-C)  
**Challenge**: Requires exactly 3 colors due to the triangle A-B-C  
**Starting Position**: Node A  
**Visibility**: Can only see immediate neighbors at each step  
**Optimal Strategy**: Plan movement to minimize reassignments while exploring efficiently


## Running and Testing Your Agent

### Testing Your Implementation
```bash
python game_runner.py
```

### Expected Tournament Output
```
Starting CSP Tournament. Agent at: A

--- Step 1 ---
Agent is at 'A'. Requesting next move...
Referee: Moved agent to 'B'. [Move Penalty: -1]
Agent is now at 'B'. Requesting color...
Referee: Assigned color 'Red' to node 'B'.

--- Step 2 ---
Agent is at 'B'. Requesting next move...
Referee: Moved agent to 'C'. [Move Penalty: -1]
Agent is now at 'C'. Requesting color...  
Referee: Assigned color 'Green' to node 'C'.

...

--- Tournament Complete! ---
==================== FINAL SUMMARY ====================
{
  "is_correct": true,
  "nodes_colored": 5,
  "total_nodes": 5,
  "moves_made": 8,
  "reassignments": 0,
  "base_score": 100,
  "move_penalty": -8,
  "reassignment_penalty": 0,
  "perfection_bonus": 10,
  "final_score": 102
}
```

## Tournament Information

🏆 **This is a competitive assignment!** After the submission deadline, all participants' agents will compete in a tournament.

### Tournament Rules
- All agents run on the same set of test levels
- Scoring is based on the official scoring system described above
- Rankings determined by total score across all levels

## Assignment Requirements Summary

✅ **Must Implement**: Backtracking CSP solver with heuristics  
✅ **Must Achieve**: Complete and valid graph coloring  
✅ **Must Optimize**: Minimize reassignments and moves for tournament ranking  

❌ **Cannot Modify**: Game engine or framework files  
❌ **Cannot Submit**: Multiple files or renamed framework files
