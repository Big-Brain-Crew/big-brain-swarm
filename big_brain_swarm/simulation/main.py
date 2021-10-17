import numpy as np
import cv2
import pdb
from big_brain_swarm.d_star_lite.d_star_lite import initDStarLite, moveAndRescan
from big_brain_swarm.d_star_lite.grid import GridWorld
from big_brain_swarm.d_star_lite.utils import coordsToStateName, stateNameToCoords
from big_brain_swarm.Dstar_lite.d_star_lite import DStarLite
from big_brain_swarm.Dstar_lite.grid import OccupancyGridMap, SLAM

import time


def add_square_obstacle(occupancy_grid, x, y, side_length):

    # Square boundaries
    x_min = int(x - side_length / 2)
    x_max = int(x + side_length / 2)
    y_min = int(y - side_length / 2)
    y_max = int(y + side_length / 2)

    # Clamp to boundaries
    x_min = np.clip(x_min, 0, occupancy_grid.shape[1])
    x_max = np.clip(x_max, 0, occupancy_grid.shape[1])
    y_min = np.clip(y_min, 0, occupancy_grid.shape[0])
    y_max = np.clip(y_max, 0, occupancy_grid.shape[0])

    # Add to occupancy grid
    for y in range(y_min, y_max):
        occupancy_grid[y, np.arange(x_min, x_max)] = 1

    return occupancy_grid


def add_robot(occupancy_grid, x, y):

    # Robot represented by square of preset size
    side_length = 25
    x_min = int(x - side_length / 2)
    x_max = int(x + side_length / 2)
    y_min = int(y - side_length / 2)
    y_max = int(y + side_length / 2)

    # Robot must start fully on the grid
    if x_min < 0 or x_max > occupancy_grid.shape[1] or y_min < 0 or y_max > occupancy_grid.shape[0]:
        print("Error: Robot starting position invalid")
        return occupancy_grid

    # Add to occupancy grid
    for y in range(y_min, y_max):
        occupancy_grid[y, np.arange(x_min, x_max)] = 2

    return occupancy_grid


def add_goal(occupancy_grid, x, y):
    """Goal is added for visualization only"""

    side_length = 5
    x_min = int(x - side_length / 2)
    x_max = int(x + side_length / 2)
    y_min = int(y - side_length / 2)
    y_max = int(y + side_length / 2)

    # Task must be fully on the grid
    if x_min < 0 or x_max > occupancy_grid.shape[1] or y_min < 0 or y_max > occupancy_grid.shape[0]:
        print("Error: Task position invalid")
        return occupancy_grid

    # Add to occupancy grid
    for y in range(y_min, y_max):
        occupancy_grid[y, np.arange(x_min, x_max)] = 3

    return occupancy_grid


def main():

    # Create occupancy grid
    occupancy_grid_width = 200
    occupancy_grid_height = 200
    occupancy_grid = np.zeros(shape=(occupancy_grid_height, occupancy_grid_width), dtype=np.uint8)

    # Add obstacles
    occupancy_grid = add_square_obstacle(occupancy_grid, 100, 100, 15)
    # occupancy_grid = add_square_obstacle(occupancy_grid, 20, 20, 70)
    occupancy_grid = add_square_obstacle(occupancy_grid, 175, 150, 35)
    occupancy_grid = add_square_obstacle(occupancy_grid, 80, 75, 25)
    occupancy_grid = add_square_obstacle(occupancy_grid, 40, 100, 60)

    grid_world_rep = occupancy_grid.copy().astype(int)
    grid_world_rep[grid_world_rep == 1] = -1
    grid_world_rep = grid_world_rep.tolist()    

    # Add bot
    robot_start_x = 30
    robot_start_y = 35
    # occupancy_grid = add_robot(occupancy_grid, robot_start_x, robot_start_y)

    # Add navigation goal
    goal_x = 190
    goal_y = 190
    occupancy_grid = add_goal(occupancy_grid, goal_x, goal_y)

    # Initialize D* grid representation
    graph = GridWorld(occupancy_grid_width, occupancy_grid_height, external_graph=grid_world_rep)
    s_start = coordsToStateName(robot_start_x, robot_start_y)
    s_goal = coordsToStateName(goal_x, goal_y)
    
    # Add robot start and goal
    graph.setStart(s_start)
    graph.setGoal(s_goal)

    
    # pdb.set_trace()
    

    k_m = 0
    queue = []
    visibility_range = 20

    graph, queue, k_m = initDStarLite(graph, queue, s_start, s_goal, k_m)
    s_current = s_start

    # Move through grid while dynamically planning path
    s_new = ""
    while s_new != "goal":
        s_new, k_m = moveAndRescan(graph, queue, s_current, visibility_range, k_m)
        if s_new != "goal":
            s_current = s_new
            pos_coords = stateNameToCoords(s_current)

        # Display occupancy grid
        occupancy_grid_bgr = cv2.cvtColor(occupancy_grid, cv2.COLOR_GRAY2BGR)
        occupancy_grid_bgr[occupancy_grid == 0] = [255, 255, 255]
        occupancy_grid_bgr[occupancy_grid == 1] = [0, 0, 0]
        occupancy_grid_bgr[occupancy_grid == 2] = [255, 0, 0]
        occupancy_grid_bgr[occupancy_grid == 3] = [0, 255, 0]
        cv2.circle(
            occupancy_grid_bgr, (int(pos_coords[0]), int(pos_coords[1])), 8, (255, 120, 255), -1
        )
        cv2.imshow("Occupancy Grid", occupancy_grid_bgr)
        cv2.waitKey(1)

        time.sleep(0.01)


if __name__ == "__main__":
    main()
