# dynamic-pathfinder

This project implements a Dynamic Pathfinding Agent capable of navigating a grid-based environment using Informed Search Algorithms.

The system supports:
Greedy Best-First Search (GBFS)
A\* Search
Manhattan & Euclidean Heuristics
Random Maze Generation
Interactive Map Editor
Dynamic Obstacle Spawning
Real-Time Path Replanning
GUI Visualization
Performance Metrics Dashboard

The goal is to simulate real-world navigation scenarios such as robotics, autonomous systems, and Mars rover pathfinding.

# Features

1- Grid Environment

User-defined grid size (Rows × Columns)
Fixed Start and Goal nodes
Adjustable obstacle density
Manual obstacle placement/removal (mouse interaction)

2- Implemented Algorithms
1️ Greedy Best-First Search (GBFS)
Evaluation Function:
f(n) = h(n)
Uses only heuristic
Fast but not optimal
Suitable for simple maps

2️ A\* Search

Evaluation Function:
f(n) = g(n) + h(n)
Uses path cost + heuristic
Optimal (with admissible & consistent heuristic)
Reliable in complex environments

- Heuristics
  Manhattan Distance
  Euclidean Distance
  Both are admissible and consistent under standard grid movement assumptions.

- Dynamic Mode
  When enabled:
  Obstacles spawn randomly during agent movement.
  If a new obstacle blocks the current path:
  The agent detects it.
  Recalculates path from current position.
  Continues movement.
  Optimized so full search reset does not happen unnecessarily.

- Real-Time Metrics

- Nodes Expanded
- Path Cost
- Execution Time (ms)

# Installation & Setup

1 Clone the Repository
git clone https://github.com/yourusername/Dynamic-Pathfinding-Agent.git
cd Dynamic-Pathfinding-Agent
2️ Install Dependencies

If using Pygame:
pip install pygame

If using Tkinter:
Tkinter is usually pre-installed with Python.

3️ Run the Program
python main.py

# How to Use

Select Grid Size.

Choose:
Algorithm (GBFS / A\*)
Heuristic (Manhattan / Euclidean)
Set Obstacle Density.

Click:
Generate Map
Start Search
Enable Dynamic Mode to test real-time replanning.
Use mouse clicks to manually add/remove walls.

# Best Case Scenario

Minimal obstacles
Heuristic aligns with actual path
GBFS performs very fast
A\* slightly slower but optimal

# Worst Case Scenario

High obstacle density
Maze-like structure

GBFS explores many unnecessary nodes

# A\* remains optimal and more stableTheoretical Concepts Used

Informed Search
Admissibility
Consistency (Triangle Inequality)
Priority Queue
Graph Search vs Tree Search
Dynamic Replanning
