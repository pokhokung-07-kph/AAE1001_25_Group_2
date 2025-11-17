# AAE1001_25_Group_2
### Group Member
Phrolova, Nathan, David, Bob, Steven, Mathew

TASK 2: Path Planning with Multiple Cost Zones

Objective:
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
