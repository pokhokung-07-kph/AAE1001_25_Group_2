"""
D* Lite grid planning

Adapted from the A* implementation by:
author: Atsushi Sakai(@Atsushi_twi)
        Nikos Kanargias (nkana@tee.gr)

D* Lite implementation based on:
"D* Lite" by Sven Koenig and Maxim Likhachev
"""

import math
import matplotlib.pyplot as plt
import heapq

show_animation = True

class State:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.parent = None
        self.g = float('inf')
        self.rhs = float('inf')
        
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
        
    def __hash__(self):
        return hash((self.x, self.y))
        
    def __lt__(self, other):
        return False

class DStarLitePlanner:
    def __init__(self, ox, oy, resolution, rr, fc_x, fc_y, tc_x, tc_y, sc_x, sc_y, Cf, Ct, n, M):
        self.resolution = resolution
        self.rr = rr
        self.min_x, self.min_y = 0, 0
        self.max_x, self.max_y = 0, 0
        self.obstacle_map = None
        self.x_width, self.y_width = 0, 0
        self.motion = self.get_motion_model()
        
        self.calc_obstacle_map(ox, oy)
        
        self.fc_x = fc_x
        self.fc_y = fc_y
        self.tc_x = tc_x
        self.tc_y = tc_y
        self.sc_x = sc_x
        self.sc_y = sc_y
        
        self.Delta_C1 = 0.2  # cost intensive area 1 modifier
        self.Delta_C2 = 1    # cost intensive area 2 modifier
        self.Delta_C3 = 0.05 # jet stream modifier
        
        self.costPerGrid = 1
        
        # Store user input parameters
        self.Cf = Cf
        self.time_cost_level = Ct
        self.n = n
        self.M = M
        
        # D* Lite specific attributes
        self.U = []  # priority queue
        self.km = 0  # accumulative cost
        self.states = {}  # dictionary containing all created states
        
        # Aircraft specific parameters (kept from original)
        self.aircraft_params = {
            'A321': {
                'delta_F': 54,
                'Ct': {'L': 10, 'M': 15, 'H': 20},
                'C': 1800,
                'p': 200
            },
            'A339': {
                'delta_F': 84,
                'Ct': {'L': 15, 'M': 21, 'H': 27},
                'C': 2000,
                'p': 300
            },
            'A359': {
                'delta_F': 90,
                'Ct': {'L': 20, 'M': 27, 'H': 34},
                'C': 2500,
                'p': 350
            }
        }

    def calc_key(self, s):
        """Calculate the key for a state."""
        k1 = min(s.g, s.rhs) + self.h_cost(self.start, s) + self.km
        k2 = min(s.g, s.rhs)
        return [k1, k2]

    def h_cost(self, s1, s2):
        """Calculate the heuristic cost between two states."""
        w = 1.0  # heuristic weight
        return w * math.hypot(s1.x - s2.x, s1.y - s2.y) * self.costPerGrid

    def get_state(self, x, y):
        """Get the state of a node if it exists, create and return new state otherwise."""
        if (x, y) not in self.states:
            self.states[(x, y)] = State(x, y)
        return self.states[(x, y)]

    def get_cost(self, s1, s2):
        """Get the cost between two adjacent states."""
        if not self.verify_node(s2):
            return float('inf')
            
        base_cost = math.hypot(s1.x - s2.x, s1.y - s2.y) * self.costPerGrid
        
        # Apply cost modifiers based on areas
        x = self.calc_grid_position(s2.x, self.min_x)
        y = self.calc_grid_position(s2.y, self.min_y)
        
        if x in self.tc_x and y in self.tc_y:
            base_cost += self.Delta_C1 * base_cost
        if x in self.fc_x and y in self.fc_y:
            base_cost += self.Delta_C2 * base_cost
        if x in self.sc_x and y in self.sc_y:
            base_cost -= self.Delta_C3 * base_cost
            
        return base_cost

    def verify_node(self, node):
        """Verify if the node is valid (within bounds and not obstacle)."""
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)

        if px < self.min_x or py < self.min_y:
            return False
        if px >= self.max_x or py >= self.max_y:
            return False
        if self.obstacle_map[node.x][node.y]:
            return False
        return True

    def update_vertex(self, u):
        """Update vertex u and its cost if it's not the goal."""
        if u != self.goal:
            min_rhs = float('inf')
            # Look at all neighbors
            for i, _ in enumerate(self.motion):
                s = self.get_state(u.x + self.motion[i][0],
                                 u.y + self.motion[i][1])
                cost = self.get_cost(u, s)
                if cost < float('inf'):
                    min_rhs = min(min_rhs, cost + s.g)
            u.rhs = min_rhs

        # Remove u from priority queue if it exists
        self.U = [(k, s) for k, s in self.U if s != u]
        heapq.heapify(self.U)

        # If u is inconsistent, add it back to the priority queue
        if u.g != u.rhs:
            heapq.heappush(self.U, (self.calc_key(u), u))

    def compute_shortest_path(self):
        """Compute the shortest path from start to goal."""
        while (len(self.U) > 0 and
               (self.U[0][0] < self.calc_key(self.start) or
                self.start.rhs > self.start.g)):
            k_old = self.U[0][0]
            u = heapq.heappop(self.U)[1]

            if k_old < self.calc_key(u):
                heapq.heappush(self.U, (self.calc_key(u), u))
            elif u.g > u.rhs:
                u.g = u.rhs
                for i, _ in enumerate(self.motion):
                    s = self.get_state(u.x + self.motion[i][0],
                                     u.y + self.motion[i][1])
                    self.update_vertex(s)
            else:
                u.g = float('inf')
                self.update_vertex(u)
                for i, _ in enumerate(self.motion):
                    s = self.get_state(u.x + self.motion[i][0],
                                     u.y + self.motion[i][1])
                    self.update_vertex(s)

    def planning(self, sx, sy, gx, gy):
        """
        D* Lite path planning

        input:
            sx: start x position [m]
            sy: start y position [m]
            gx: goal x position [m]
            gy: goal y position [m]

        output:
            rx: x position list of the final path
            ry: y position list of the final path
        """

        # Initialize start and goal states
        self.start = self.get_state(self.calc_xy_index(sx, self.min_x),
                                  self.calc_xy_index(sy, self.min_y))
        self.goal = self.get_state(self.calc_xy_index(gx, self.min_x),
                                 self.calc_xy_index(gy, self.min_y))

        # Initialize goal state
        self.goal.rhs = 0
        heapq.heappush(self.U, (self.calc_key(self.goal), self.goal))

        self.compute_shortest_path()

        # Extract path
        rx, ry = [], []
        current = self.start

        while current != self.goal:
            # Find best neighbor
            min_cost = float('inf')
            best_next = None
            
            for i, _ in enumerate(self.motion):
                next_state = self.get_state(current.x + self.motion[i][0],
                                          current.y + self.motion[i][1])
                cost = self.get_cost(current, next_state)
                if cost < float('inf'):
                    total_cost = cost + next_state.g
                    if total_cost < min_cost:
                        min_cost = total_cost
                        best_next = next_state

            if best_next is None:
                print("No path found")
                break

            current = best_next
            rx.append(self.calc_grid_position(current.x, self.min_x))
            ry.append(self.calc_grid_position(current.y, self.min_y))

            if show_animation:
                plt.plot(rx[-1], ry[-1], "xc")
                plt.pause(0.01)

        return rx, ry

    def calc_obstacle_map(self, ox, oy):
        """Calculate obstacle map."""
        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))

        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)

        self.obstacle_map = [[False for _ in range(self.y_width)]
                           for _ in range(self.x_width)]
        for ix in range(self.x_width):
            x = self.calc_grid_position(ix, self.min_x)
            for iy in range(self.y_width):
                y = self.calc_grid_position(iy, self.min_y)
                for iox, ioy in zip(ox, oy):
                    d = math.hypot(iox - x, ioy - y)
                    if d <= self.rr:
                        self.obstacle_map[ix][iy] = True
                        break

    @staticmethod
    def get_motion_model():
        """Get motion model."""
        motion = [[1, 0, 1],
                 [0, 1, 1],
                 [-1, 0, 1],
                 [0, -1, 1],
                 [-1, -1, math.sqrt(2)],
                 [-1, 1, math.sqrt(2)],
                 [1, -1, math.sqrt(2)],
                 [1, 1, math.sqrt(2)]]
        return motion

    def calc_grid_position(self, index, min_position):
        """Calculate grid position."""
        pos = index * self.resolution + min_position
        return pos

    def calc_xy_index(self, position, min_pos):
        """Calculate x,y index."""
        return round((position - min_pos) / self.resolution)

def main():
    print("D* Lite path planning demo")

    # Get user inputs
    Cf = float(input("Enter the cost of fuel (Cf): "))
    
    while True:
        time_cost = input("Enter time cost (L/M/H): ").upper()
        if time_cost in ['L', 'M', 'H']:
            Ct = {'L': 0.8, 'M': 1.0, 'H': 1.2}[time_cost]
            break
        print("Invalid input. Please enter L, M, or H.")
    
    while True:
        try:
            n = int(input("Enter number of passengers: "))
            if n > 0:
                break
            print("Number of passengers must be positive.")
        except ValueError:
            print("Please enter a valid integer.")
    
    while True:
        try:
            M = int(input("Enter maximum number of flights: "))
            if M > 0:
                break
            print("Maximum number of flights must be positive.")
        except ValueError:
            print("Please enter a valid integer.")

    # start and goal position
    sx = 0.0
    sy = 10.0
    gx = 60.0
    gy = 25.0
    grid_size = 1
    robot_radius = 1.0

    # set obstacle positions
    ox, oy = [], []
    for i in range(-10, 70):  # bottom border
        ox.append(i)
        oy.append(-10.0)
    for i in range(-10, 60):  # right border
        ox.append(70.0)
        oy.append(i)
    for i in range(-10, 70):  # top border
        ox.append(i)
        oy.append(60.0)
    for i in range(-10, 60):  # left border
        ox.append(-10.0)
        oy.append(i)

    for i in range(0, 20):  # internal obstacles
        ox.append(20.0)
        oy.append(i)
    
    for i in range(30, 55):
        ox.append(10.0)
        oy.append(i)
    
    for i in range(0, 20):
        ox.append(30.0)
        oy.append(i)

    # set cost intensive areas
    tc_x, tc_y = [], []
    for i in range(10, 20):
        for j in range(10, 45):
            tc_x.append(i)
            tc_y.append(j)
    
    fc_x, fc_y = [], []
    for i in range(30, 45):
        for j in range(10, 35):
            fc_x.append(i)
            fc_y.append(j)

    # set jet stream
    sc_x, sc_y = [], []
    for i in range(0, 60):
        for j in range(-2, 3):
            sc_x.append(i)
            sc_y.append(j)

    if show_animation:
        plt.plot(ox, oy, ".k")  # plot obstacles
        plt.plot(sx, sy, "og")  # plot start
        plt.plot(gx, gy, "xb")  # plot goal
        plt.plot(fc_x, fc_y, "oy")  # plot cost area 1
        plt.plot(tc_x, tc_y, "or")  # plot cost area 2
        plt.plot(sc_x, sc_y, "og")  # plot jet stream
        plt.grid(True)
        plt.axis("equal")

    dstar = DStarLitePlanner(ox, oy, grid_size, robot_radius, 
                            fc_x, fc_y, tc_x, tc_y, sc_x, sc_y,
                            Cf, Ct, n, M)
    rx, ry = dstar.planning(sx, sy, gx, gy)

    if show_animation:
        plt.plot(rx, ry, "-r")  # show final path
        plt.pause(0.001)
        plt.show()

if __name__ == '__main__':
    main()