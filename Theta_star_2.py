import math
import matplotlib.pyplot as plt

show_animation = True


class ThetaStarPlanner:

    def __init__(self, ox, oy, resolution, rr, fc_x, fc_y, tc_x, tc_y, sc_x, sc_y, Cf, Ct, n, M):
        """
        Initialize grid map for Theta* planning

        ox: x position list of Obstacles [m]
        oy: y position list of Obstacles [m]
        resolution: grid resolution [m]
        rr: robot radius[m]
        Cf: cost of fuel
        Ct: time cost factor (L=0.8, M=1.0, H=1.2)
        n: number of passengers
        M: maximum number of flights
        """

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

        self.Delta_C1 = 0.2
        self.Delta_C2 = 1
        self.Delta_C3 = 0.05

        self.costPerGrid = 1

        # Store user input parameters
        self.Cf = Cf
        self.time_cost_level = Ct
        self.n = n
        self.M = M

        # Aircraft specific parameters
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

    class Node:
        def __init__(self, x, y, cost, parent_index):
            self.x = x
            self.y = y
            self.cost = cost
            self.parent_index = parent_index

        def __str__(self):
            return f"{self.x},{self.y},{self.cost},{self.parent_index}"

    def planning(self, sx, sy, gx, gy):
        """
        Theta* path search

        returns rx, ry (lists of positions)
        """

        start_node = self.Node(self.calc_xy_index(sx, self.min_x),
                               self.calc_xy_index(sy, self.min_y), 0.0, -1)
        goal_node = self.Node(self.calc_xy_index(gx, self.min_x),
                              self.calc_xy_index(gy, self.min_y), float("inf"), -1)

        open_set, closed_set = dict(), dict()
        start_index = self.calc_grid_index(start_node)
        open_set[start_index] = start_node

        while True:
            if len(open_set) == 0:
                print("Open set is empty..")
                break

            # pick node with smallest f = g + h
            c_id = min(open_set, key=lambda o: open_set[o].cost + self.calc_heuristic(open_set[o], goal_node))
            current = open_set[c_id]

            # show graph
            if show_animation:
                plt.plot(self.calc_grid_position(current.x, self.min_x),
                         self.calc_grid_position(current.y, self.min_y), "xc")
                plt.gcf().canvas.mpl_connect('key_release_event',
                                             lambda event: [exit(0) if event.key == 'escape' else None])
                if len(closed_set.keys()) % 10 == 0:
                    plt.pause(0.001)

            # reached goal (grid index match)
            if current.x == goal_node.x and current.y == goal_node.y:
                Tbest = current.cost
                print("\nTotal Trip time required -> ", Tbest)

                # Get time cost level (L/M/H)
                time_cost_level = 'M'
                if self.time_cost_level == 0.8:
                    time_cost_level = 'L'
                elif self.time_cost_level == 1.2:
                    time_cost_level = 'H'

                print("\nFlight Analysis for each aircraft type:")
                print("-" * 60)
                print("Aircraft | Flight   | Total Cost    | Ultimate")
                print("Type    | Demand   | per Flight    | Cost")
                print("-" * 60)

                viable_options = {}

                for aircraft in ['A321', 'A339', 'A359']:
                    params = self.aircraft_params[aircraft]
                    delta_F = params['delta_F']
                    Ct = params['Ct'][time_cost_level]
                    C = params['C']
                    p = params['p']

                    d = math.ceil(self.n / p)

                    if d <= self.M:
                        total_cost = self.Cf * delta_F * Tbest + Ct * Tbest + C
                        ultimate_cost = total_cost * d
                        viable_options[aircraft] = ultimate_cost
                        print(f"{aircraft:8} | {d:8} | {total_cost:12,.2f} | {ultimate_cost:,.2f}")
                    else:
                        print(f"{aircraft:8} | {'n/a':8} | {'n/a':12} | {'n/a':>12}")

                print("-" * 60)

                if viable_options:
                    best_aircraft = min(viable_options.items(), key=lambda x: x[1])
                    print(f"\nOptimal Choice:")
                    print(f"Aircraft Type: {best_aircraft[0]}")
                    print(f"Ultimate Cost: {best_aircraft[1]:,.2f}")
                else:
                    print("\nNo viable aircraft options available for the given constraints.")

                goal_node.parent_index = current.parent_index
                goal_node.cost = current.cost
                break

            # move current from open to closed
            del open_set[c_id]
            closed_set[c_id] = current

            # expand neighbors
            for i, _ in enumerate(self.motion):
                nx = current.x + self.motion[i][0]
                ny = current.y + self.motion[i][1]
                step_cost = self.motion[i][2] * self.costPerGrid

                node = self.Node(nx, ny, current.cost + step_cost, c_id)

                # add cost area modifiers
                if self.calc_grid_position(node.x, self.min_x) in self.tc_x:
                    if self.calc_grid_position(node.y, self.min_y) in self.tc_y:
                        node.cost += self.Delta_C1 * self.motion[i][2]
                if self.calc_grid_position(node.x, self.min_x) in self.fc_x:
                    if self.calc_grid_position(node.y, self.min_y) in self.fc_y:
                        node.cost += self.Delta_C2 * self.motion[i][2]
                if self.calc_grid_position(node.x, self.min_x) in self.sc_x:
                    if self.calc_grid_position(node.y, self.min_y) in self.sc_y:
                        node.cost -= self.Delta_C3 * self.motion[i][2]

                n_id = self.calc_grid_index(node)

                if not self.verify_node(node):
                    continue

                if n_id in closed_set:
                    continue

                # Theta* specific relaxation:
                # Try to connect neighbor to the parent of current (if exists) via line-of-sight
                use_parent = False
                if current.parent_index != -1:
                    parent = closed_set.get(current.parent_index, None)
                    if parent is not None:
                        if self.line_of_sight(parent, node):
                            # cost via parent's position to neighbor (euclidean)
                            parent_pos_cost = parent.cost + math.hypot(
                                self.calc_grid_position(parent.x, self.min_x) - self.calc_grid_position(node.x, self.min_x),
                                self.calc_grid_position(parent.y, self.min_y) - self.calc_grid_position(node.y, self.min_y)
                            ) * self.costPerGrid
                            if parent_pos_cost + 1e-6 < node.cost:
                                node.cost = parent_pos_cost
                                node.parent_index = self.calc_grid_index(parent)
                                use_parent = True

                # If not using parent relaxation, we already set node.cost = current.cost + step_cost and parent = current
                if not use_parent:
                    node.parent_index = c_id

                # discovered new node or found a better path?
                if n_id not in open_set:
                    open_set[n_id] = node
                else:
                    if open_set[n_id].cost > node.cost:
                        open_set[n_id] = node

        rx, ry = self.calc_final_path(goal_node, closed_set)
        return rx, ry

    def calc_final_path(self, goal_node, closed_set):
        rx = [self.calc_grid_position(goal_node.x, self.min_x)]
        ry = [self.calc_grid_position(goal_node.y, self.min_y)]
        parent_index = goal_node.parent_index
        # if the parent path includes nodes in closed_set, follow them
        while parent_index != -1 and parent_index in closed_set:
            n = closed_set[parent_index]
            rx.append(self.calc_grid_position(n.x, self.min_x))
            ry.append(self.calc_grid_position(n.y, self.min_y))
            parent_index = n.parent_index
        return rx, ry

    def calc_heuristic(self, n1, n2):
        w = 1.0
        d = w * math.hypot(n1.x - n2.x, n1.y - n2.y)
        return d * self.costPerGrid

    def calc_grid_position(self, index, min_position):
        pos = index * self.resolution + min_position
        return pos

    def calc_xy_index(self, position, min_pos):
        return round((position - min_pos) / self.resolution)

    def calc_grid_index(self, node):
        return (node.y - self.min_y) * self.x_width + (node.x - self.min_x)

    def verify_node(self, node):
        px = self.calc_grid_position(node.x, self.min_x)
        py = self.calc_grid_position(node.y, self.min_y)

        if px < self.min_x:
            return False
        elif py < self.min_y:
            return False
        elif px >= self.max_x:
            return False
        elif py >= self.max_y:
            return False

        # collision check
        # ensure indices are in bounds
        if node.x < 0 or node.x >= self.x_width or node.y < 0 or node.y >= self.y_width:
            return False

        if self.obstacle_map[node.x][node.y]:
            return False

        return True

    def calc_obstacle_map(self, ox, oy):
        self.min_x = round(min(ox))
        self.min_y = round(min(oy))
        self.max_x = round(max(ox))
        self.max_y = round(max(oy))
        print("min_x:", self.min_x)
        print("min_y:", self.min_y)
        print("max_x:", self.max_x)
        print("max_y:", self.max_y)

        self.x_width = round((self.max_x - self.min_x) / self.resolution)
        self.y_width = round((self.max_y - self.min_y) / self.resolution)
        print("x_width:", self.x_width)
        print("y_width:", self.y_width)

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

    def line_of_sight(self, node1, node2):
        """
        Check line-of-sight between node1 and node2.
        We sample along the straight line at a small step and check for collisions.
        node1 and node2 are Node objects with grid indices.
        """
        x1 = self.calc_grid_position(node1.x, self.min_x)
        y1 = self.calc_grid_position(node1.y, self.min_y)
        x2 = self.calc_grid_position(node2.x, self.min_x)
        y2 = self.calc_grid_position(node2.y, self.min_y)

        dist = math.hypot(x2 - x1, y2 - y1)
        if dist == 0:
            return True

        # sample step: a fraction of resolution for safety
        step = self.resolution * 0.5
        steps = int(math.ceil(dist / step))

        for i in range(steps + 1):
            t = i / steps
            xs = x1 + (x2 - x1) * t
            ys = y1 + (y2 - y1) * t
            ix = self.calc_xy_index(xs, self.min_x)
            iy = self.calc_xy_index(ys, self.min_y)

            if ix < 0 or ix >= self.x_width or iy < 0 or iy >= self.y_width:
                return False
            if self.obstacle_map[ix][iy]:
                return False
        return True

    @staticmethod
    def get_motion_model():
        motion = [[1, 0, 1],
                  [0, 1, 1],
                  [-1, 0, 1],
                  [0, -1, 1],
                  [-1, -1, math.sqrt(2)],
                  [-1, 1, math.sqrt(2)],
                  [1, -1, math.sqrt(2)],
                  [1, 1, math.sqrt(2)]]
        return motion


def main():
    print(__file__ + " start the Theta* algorithm demo !!")

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

    sx = 0.0
    sy = 10.0
    gx = 60.0
    gy = 25.0
    grid_size = 1
    robot_radius = 1.0

    # obstacles (group 9 example)
    ox, oy = [], []
    for i in range(-10, 70):
        ox.append(i)
        oy.append(-10.0)
    for i in range(-10, 60):
        ox.append(70.0)
        oy.append(i)
    for i in range(-10, 70):
        ox.append(i)
        oy.append(60.0)
    for i in range(-10, 60):
        ox.append(-10.0)
        oy.append(i)

    for i in range(0, 20):
        ox.append(20.0)
        oy.append(i)

    for i in range(30, 55):
        ox.append(10.0)
        oy.append(i)

    for i in range(0, 20):
        ox.append(30.0)
        oy.append(i)

    # cost intensive areas
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

    sc_x, sc_y = [], []
    for i in range(0, 60):
        for j in range(-2, 3):
            sc_x.append(i)
            sc_y.append(j)

    if show_animation:
        plt.plot(ox, oy, ".k")
        plt.plot(sx, sy, "og")
        plt.plot(gx, gy, "xb")

        plt.plot(fc_x, fc_y, "oy")
        plt.plot(tc_x, tc_y, "or")
        plt.plot(sc_x, sc_y, "og")

        plt.grid(True)
        plt.axis("equal")

    planner = ThetaStarPlanner(ox, oy, grid_size, robot_radius, fc_x, fc_y, tc_x, tc_y, sc_x, sc_y, Cf, Ct, n, M)
    rx, ry = planner.planning(sx, sy, gx, gy)

    if show_animation:
        plt.plot(rx, ry, "-r")
        plt.pause(0.001)
        plt.show()


if __name__ == '__main__':
    main()
