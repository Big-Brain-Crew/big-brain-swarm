import numpy as np
import cv2
import pdb
from big_brain_swarm.d_star_lite.d_star_lite import initDStarLite, moveAndRescan
from big_brain_swarm.d_star_lite.grid import GridWorld
from big_brain_swarm.d_star_lite.utils import coordsToStateName, stateNameToCoords

import time


class OccupancyGrid:
    class RobotObstacle:
        def __init__(self, id, x, y, side_length):
            self.id = id
            self.side_length = side_length
            self.update(x, y)

        def update(self, x, y):
            self.x = x
            self.y = y

            # Robot represented as a square obstacle of preset size
            self.x_min = int(x - self.side_length / 2)
            self.x_max = int(x + self.side_length / 2)
            self.y_min = int(y - self.side_length / 2)
            self.y_max = int(y + self.side_length / 2)

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.grid = np.zeros(shape=(self.height, self.width), dtype=np.uint8)

        self.robots = []

    def add_square_obstacle(self, x, y, side_length):

        # Square boundaries
        x_min = int(x - side_length / 2)
        x_max = int(x + side_length / 2)
        y_min = int(y - side_length / 2)
        y_max = int(y + side_length / 2)

        # Clamp to boundaries
        x_min = np.clip(x_min, 0, self.width)
        x_max = np.clip(x_max, 0, self.width)
        y_min = np.clip(y_min, 0, self.height)
        y_max = np.clip(y_max, 0, self.height)

        # Add to occupancy grid
        for y in range(y_min, y_max):
            self.grid[y, np.arange(x_min, x_max)] = 1

    def add_robot(self, id, x, y):
        side_length = 25
        robot = self.RobotObstacle(id, x, y, side_length)

        # Robot must start fully on the grid
        robot.x_min = np.clip(robot.x_min, 0, self.width)
        robot.x_max = np.clip(robot.x_max, 0, self.width)
        robot.y_min = np.clip(robot.y_min, 0, self.height)
        robot.y_max = np.clip(robot.y_max, 0, self.height)

        # Add to occupancy grid
        # Offset of 2 required since 0 (free) and 1 (obstacle) are taken
        for y in range(robot.y_min, robot.y_max):
            self.grid[y, np.arange(robot.x_min, robot.x_max)] = id + 2

        self.robots.append(robot)

    def update_robot_position(self, id, x, y):
        """
        NOTE: This could be made more efficient by keeping values that overlap
        between old and new obstacle boundaries. But I will skip that for now.
        """
        robot = self.robots[id]

        # Remove old obstacle
        for y in range(robot.y_min, robot.y_max):
            self.grid[y, np.arange(robot.x_min, robot.x_max)] = 0

        robot.update(x, y)

        # Robot must start fully on the grid
        robot.x_min = np.clip(robot.x_min, 0, self.width)
        robot.x_max = np.clip(robot.x_max, 0, self.width)
        robot.y_min = np.clip(robot.y_min, 0, self.height)
        robot.y_max = np.clip(robot.y_max, 0, self.height)        

        # Place new obstacle
        # print(robot.x_min, robot.x_max, robot.y_min, robot.y_max)
        for y in range(robot.y_min, robot.y_max):
            self.grid[y, np.arange(robot.x_min, robot.x_max)] = id + 2

    def add_goal(self, x, y):
        """Goal is added for visualization only"""

        side_length = 5
        x_min = int(x - side_length / 2)
        x_max = int(x + side_length / 2)
        y_min = int(y - side_length / 2)
        y_max = int(y + side_length / 2)

        # Task must be fully on the grid
        if x_min < 0 or x_max > self.width or y_min < 0 or y_max > self.height:
            print("Error: Task position invalid")
            return

        # Add to occupancy grid
        for y in range(y_min, y_max):
            self.grid[y, np.arange(x_min, x_max)] = 3

    def convert_to_grid_world_list(self, id):
        grid_world_rep = self.grid.copy().astype(int)

        # Remove self from grid
        grid_world_rep[grid_world_rep == id + 2] = 0

        # Convert obstacles and other robots
        grid_world_rep[grid_world_rep > 0] = -1

        # Convert to list
        grid_world_rep = grid_world_rep.tolist()


class SwarmBotState:
    def __init__(
        self, id, start_x=None, start_y=None, goal_x=None, goal_y=None, visibility_range=5
    ):
        self.id = id
        self.start_x = start_x
        self.start_y = start_y
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.visibility_range = visibility_range

        self.graph = None
        self.k_m = 0
        self.queue = []
        self.s_current = None
        self.s_new = ""


def main():

    # Create occupancy grid
    occupancy_grid = OccupancyGrid(200, 200)

    # Add obstacles
    occupancy_grid.add_square_obstacle(100, 100, 15)
    occupancy_grid.add_square_obstacle(175, 150, 35)
    occupancy_grid.add_square_obstacle(80, 75, 25)
    occupancy_grid.add_square_obstacle(40, 100, 80)

    visibility_range = 20
    swarm_state = [
        SwarmBotState(
            id=0, start_x=30, start_y=35, goal_x=150, goal_y=35, visibility_range=visibility_range
        ),
        SwarmBotState(
            id=1, start_x=150, start_y=35, goal_x=30, goal_y=35, visibility_range=visibility_range
        ),
    ]

    # Initialize robot path planning states
    for s in swarm_state:

        # Add navigation goal for visualization
        occupancy_grid.add_goal(s.goal_x, s.goal_y)

        # Add robots to occupancy grid
        occupancy_grid.add_robot(s.id, s.start_x, s.start_y)

        # Convert occupancy grid to GridWorld
        grid_world_rep = occupancy_grid.convert_to_grid_world_list(s.id)

        # Initialize D* grid representation
        graph = GridWorld(
            occupancy_grid.width, occupancy_grid.height, external_graph=grid_world_rep
        )

        # Add robot start and goal
        s_start = coordsToStateName(s.start_x, s.start_y)
        s_goal = coordsToStateName(s.goal_x, s.goal_y)
        graph.setStart(s_start)
        graph.setGoal(s_goal)

        # Add to swarm state
        s.graph = graph

        # Initiliaze D*
        s.graph, s.queue, s.k_m = initDStarLite(s.graph, s.queue, s_start, s_goal, s.k_m)
        s.s_current = s_start

    # Move through grid while dynamically planning path
    while True:
        swarm_coordinates = []
        robot_position_updates = []
        for s in swarm_state:
            s.s_new, s.k_m = moveAndRescan(s.graph, s.queue, s.s_current, s.visibility_range, s.k_m)
            if s.s_new != "goal":

                # Add to list of robot updates
                current_x, current_y = stateNameToCoords(s.s_current)
                robot_position_update = {
                    "id": s.id,
                    "x": current_x, 
                    "y": current_y,
                    "action" : "remove"
                    }
                robot_position_updates.append(robot_position_update)

                # Send new position to robots
                # This is where you will send stuff over comms
                s.s_current = s.s_new

                # Update the robot "obstacle" in occupancy grid
                current_x, current_y = stateNameToCoords(s.s_current)
                occupancy_grid.update_robot_position(s.id, current_x, current_y)

                # Add to list of robot updates
                robot_position_update = {
                    "id": s.id,
                    "x": current_x, 
                    "y": current_y,
                    "action" : "add"
                    }
                robot_position_updates.append(robot_position_update)

                # For displaying position
                swarm_coordinates.append([current_x, current_y])
            
        # Update GridWorlds with new robot positions
        for s in swarm_state:
            for robot in robot_position_updates:
                if robot["id"] != s.id:
                    if robot["action"] == "remove":
                        s.graph.removeRobotObstacle(occupancy_grid.robots[robot["id"]])
                    else:
                        s.graph.addRobotObstacle(occupancy_grid.robots[robot["id"]])
                    

        # Display occupancy grid
        occupancy_grid_bgr = cv2.cvtColor(occupancy_grid.grid, cv2.COLOR_GRAY2BGR)
        occupancy_grid_bgr[occupancy_grid == 0] = [255, 255, 255]
        occupancy_grid_bgr[occupancy_grid == 1] = [0, 0, 0]
        occupancy_grid_bgr[occupancy_grid == 2] = [255, 0, 0]
        occupancy_grid_bgr[occupancy_grid == 3] = [0, 255, 0]

        color_increment = 100
        for coords in swarm_coordinates:
            cv2.circle(
                occupancy_grid_bgr,
                (int(coords[0]), int(coords[1])),
                8,
                (255, 120+color_increment, 255),
                -1,
            )
        cv2.imshow("Occupancy Grid", occupancy_grid_bgr)
        cv2.waitKey(1)

        time.sleep(0.01)


if __name__ == "__main__":
    main()
