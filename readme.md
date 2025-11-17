# AAE1001_25_Group_2
### Group Member:

*Phrolova, Nathan, David, Bob, Steven, Matthew



# Task 1
## Table of Contents

- [Background](#background)
- [Scenarios](#scenarios)
  - [Scenario 1: Medium Passengers](#scenario-1-medium-passengers)
  - [Scenario 2: High Passengers](#scenario-2-high-passengers)
  - [Scenario 3: Low Passengers](#scenario-3-low-passengers)

## Background

- **Start node:** E60, 25
- **Goal node:** 30, 20
- **Key areas:** Fuel-consuming, Time-consuming
<img width="862" height="483" alt="Screenshot 2025-11-17 at 18 02 54" src="https://github.com/user-attachments/assets/a9e7a61f-9424-4dcc-a985-57b4fc95eab1" />

## Scenarios

### Scenario 1: 

- **Fuel Cost:** 0.85
- **Time Cost:** Medium
- **Number of Passengers:** 330
- **Maximum Flight:** 13

**Aircraft Costs:**
| Aircraft Type | Demand per Flight | Total Cost  |
|---------------|------------------|-------------|
| A330-900neo          | 11               | 99,403.26   |
| A350-900         | 10               | 103,819.73  |

A321neo do not meet the requirement in this situation.

**Optimal Aircraft:** **A330-900neo**  
**Cost:** 99,403.26

### Scenario 2: 

- **Fuel Cost:** 0.96
- **Time Cost:** High
- **Number of Passengers:** 1500
- **Maximum Flight:** 7

**Aircraft Costs:**
| Aircraft Type | Demand per Flight | Total Cost  |
|---------------|------------------|-------------|
| A330-900neo          | 5                | 50,986.26   |
| A350-900          | 5                | 58,344.91   |

A321neo do not meet the requirement in this situation.

**Optimal Aircraft:** **A330-900neo**  
**Cost:** 50,986.26

### Scenario 3: 

- **Fuel Cost:** 0.78
- **Time Cost:** Low
- **Number of Passengers:** 2250
- **Maximum Flight:** 25

**Aircraft Costs:**
| Aircraft Type | Demand per Flight | Total Cost  |
|---------------|------------------|-------------|
| A321neo          | 12               | 69,229.96   |
| A330-900neo          | 8                | 65,055.57   |
| A350-900          | 7                | 65,583.84   |

**Optimal Aircraft:** **A330-900neo**  
**Cost:** 65,055.57


<p align="center">

  <h1 align="center"> TASK 2: 
<p align="center">

  <h3 align="center"> Path Planning with Multiple Cost Zones </h3>

## Objective:
To implement an A* path planning algorithm that accounts for multiple types of cost-intensive areas:

Fuel-consuming areas (yellow)

Time-consuming areas (red)

Newly defined special-cost areas (green)

Implementation:

Extended the AStarPlanner class to incorporate three cost modifiers:

Delta_C1 = 0.2 for fuel-consuming zones

Delta_C2 = 1 for time-consuming zones

Delta_C3 = 0.05 for special-cost zones

The algorithm adjusts node cost when passing through these zones

Visualization includes plotting all three zones in different colors

Created a new special-cost area defined by coordinates and plotted in green

Results:

Final Route Time: 74.46793283 units

The path successfully avoids high-cost regions while optimizing for both time and fuel

The algorithm efficiently navigates through multiple cost-intensive areas


TASK 2A: Restricted Movement Model

Objective:
To modify the A* algorithm to restrict movement to only horizontal and vertical directions (no diagonals).

Implementation:

Updated the motion model in AStarPlanner to allow only:

[1, 0, 1] → Right

[0, 1, 1] → Up

[-1, 0, 1] → Left

[0, -1, 1] → Down

This change ensures the robot moves only in four cardinal directions

Simplified the path planning by eliminating diagonal movements

Results:

The final path is more constrained but remains optimal under movement restrictions

Useful for scenarios where diagonal movement is impractical or unsafe

Maintained efficient path planning while adhering to movement constraints







<p align="center">

  <h1 align="center"> Task A3
<p align="center">

  <h3 align="center"> Path Planning Algorithms Comparison </h3>

This task conducts a comparative analysis of three path planning algorithms—A*, D* lite, and Theta*—to determine their performance in cost-sensitive environments with obstacles. The mission is to identify which algorithm provides the optimal balance of speed, reliability, and cost-efficiency for robotic navigation and autonomous systems.
This project includes implementations and visualizations to demonstrate their performance under an identical scenario.

# Map Setup #
- *Fixed start and end points for consistent comparison*
- *Black line obstacles that must be avoided*
- *Cost-varying terrain yellow/red areas increase movement cost, green areas decrease cost*

# Methodology
- *Same environmental constraints and cost parameters*
- *Identical start-goal configurations*
- *Standardized performance metrics: path validity, computation speed, and cost optimization*

# Algorithms Overview ##

### 1. A* (A-Star) Algorithm
A* is a popular pathfinding algorithm that finds the shortest path from a start to a goal. It smartly balances exploring the map and heading toward the goal using a simple guess (heuristic).

**Key Characteristics:**
- **Completeness**: Guaranteed to find a path if one exists
- **Optimality**: Guaranteed to find the shortest path when using an admissible heuristic
- **Speed**: O(b^d) where b is branching factor and d is solution depth
- **The Basic Finder**: Finds the shortest path on a grid.
  
### 2. D* (Dynamic A-Star) Algorithm
D* is a smart version of A* that quickly updates the path when the map changes. It is an incremental search algorithm that is particularly efficient for replanning when the environment changes or new obstacles are discovered. It's especially useful in robotics and dynamic environments.

**Key Characteristics:** 
- **Incremental**: Efficiently replans when the environment changes
- **Lifelong Planning**: Can handle dynamic obstacles
- **Backward Search**: Processes from the goal back to the start
- **Better for Dynamic Environments**: More efficient than replanning with A* from scratch

### 3. Theta* Algorithm
Theta* is an any-angle path planning algorithm that builds upon A* but allows paths to propagate between nodes without constraining movement to grid edges, resulting in more direct and realistic paths.

**Key Characteristics:**
- **Any-Angle Paths**: Creates smoother, more direct paths
- **Line-of-Sight Checking**: Uses ray casting to find direct paths between non-adjacent nodes
- **Shorter Paths**: Typically finds shorter paths than grid-constrained algorithms
- **More Realistic**: Better approximates true shortest paths in continuous space

##

# Results Summary

| Algorithm | Path Success & Quality | Computational Speed | Key Reason |
|-----------|------------------------|---------------------|------------|
| **D***    | **Success** - Found optimal cost-effective path | **Fastest** | Reverse search + efficient cost propagation |
| **A***    | **Success** - Found same path as D* | **Slower** | Reliable grid-search, less optimized than D* |
| **Theta*** | **Success** - Found shortest, the most cost-effective path |**Fastest** | Any-angle paths reducing Euclidean distance and travel time.|

# Path Planning Algorithms Performance Comparison

### A* Algorithm Performance
![A* Algorithm](Task_3A/img/a_star_demo.gif)

*Computational time* → ≈15s

*Path length* → ≈74.89

*Description: A* demonstrated solid performance, serving as a reliable baseline algorithm. It successfully found the shortest and valid path while avoiding all obstacles. However, it operated at medium speed - slower than D* due to its forward search approach and expanding unecessary nodes. Thanks to its admissible heuristic and proper cost accumulation in the g-cost, it considers terrain costs while guaranteeing the optimal path, finding the most economical solution*

### D* Algorithm Performance  
![D* Algorithm](Task_3A/img/d_star_demo.gif)

*Computational Time* → ≈5s

*Path length* → ≈74.89

*Description: D* demonstrated competitive performance in the path planning task. It successfully found the shortest and valid path while avoiding all obstacles and showing exceptional speed in the environments. This is because it uses a reverse search approach from goal to start and employs dynamic cost propagation which allows it to expand fewer nodes than A*. Because of its incremental search and cost-updating mechanism, it intelligently considers terrain costs while guaranteeing optimal paths, finding the most economical solution with minimal computational overhead. It's an extension of the A* search algorithm, so its ability to cache cost information and prioritize nodes based on key modifiers enables faster convergence than traditional graph search methods.

### Theta* Algorithm Performance
![Theta* Algorithm](Task_3A/img/theta_star_demo.gif)

*Computational Time* → ≈2s

*Path length* → ≈63.07

*Description: Theta* demonstrated the best performance in this task. It successfully generated a shorter trajectory than both A* and D* , achieving the lowest time cost despite potential terrain penalties. This is because it uses any-angle path planning with line-of-sight checks, which allows it to bypass grid constraints and create more direct paths between points. Due to its geometric optimization and path smoothing capabilities, it minimizes overall travel distance even if it occasionally traverses higher-cost areas. It's an extension of the A search algorithm, so its ability to connect non-adjacent nodes through straight-line paths enables more natural and efficient trajectories than traditional grid-based search methods, though this comes with potential trade-offs in obstacle avoidance reliability. *

# Key Difference 

## D* and A* path

![d_star_demo.png](Task_3A/img/A&D.png)
* The diagonal movement of D* and A* are restricted to a single grid 

## Theta* path

![theta_star_demo.png](Task_3A/img/thetaandstar_demo.png)
* Theta* can perfom diagonal motion without limiting to grids, moving in any direction.


# Recommendations

- **D\***: Best for cost-sensitive environments with obstacles
- **A\***: Reliable baseline for simple path planning
- **Theta\***: Preferred for time-critical applications where path smoothness reduces travel time


