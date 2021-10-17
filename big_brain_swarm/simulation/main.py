import numpy as np
import cv2
import pdb
from big_brain_swarm.d_star_lite.d_star_lite import initDStarLite, moveAndRescan
from big_brain_swarm.d_star_lite.grid import GridWorld
from big_brain_swarm.d_star_lite.utils import coordsToStateName, stateNameToCoords
import pygame

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
        side_length = 2
        robot = self.RobotObstacle(id, x, y, side_length)
        print("Adding robot with id at", id, x, y)

        # Robot must start fully on the grid
        robot.x_min = np.clip(robot.x_min, 0, self.width)
        robot.x_max = np.clip(robot.x_max, 0, self.width)
        robot.y_min = np.clip(robot.y_min, 0, self.height)
        robot.y_max = np.clip(robot.y_max, 0, self.height)
        print(id, robot.x_min, robot.x_max, robot.y_min, robot.y_max)

        # Add to occupancy grid
        # Offset of 2 required since 0 (free) and 1 (obstacle) are taken
        for y in range(robot.y_min, robot.y_max):
            self.grid[y, np.arange(robot.x_min, robot.x_max)] = id + 2

        self.robots.append(robot)

    def update_robot_position(self, id, x, y):
        """
        NOTE: This could be made more efficient by keeping values that overlap
        between old and new obstacle boundaries.
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
        # pdb.set_trace()
        grid_world_rep[grid_world_rep == (id + 2)] = 0

        # Convert obstacles and other robots
        grid_world_rep[grid_world_rep > 0] = -1

        # Convert to list
        # pdb.set_trace()
        grid_world_rep = grid_world_rep.tolist()

        return grid_world_rep


class SwarmBotState:
    def __init__(self, id, pos_x=None, pos_y=None, goal_x=None, goal_y=None, visibility_range=5):
        self.id = id
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.goal_x = goal_x
        self.goal_y = goal_y
        self.visibility_range = visibility_range

        self.graph = None
        self.k_m = 0
        self.queue = []
        self.s_current = None
        self.s_new = ""

    def update_gridworld(self, og):
        grid_world_rep = og.convert_to_grid_world_list(self.id)
        self.graph.update_graph(grid_world_rep)


# Define some colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
GRAY1 = (145, 145, 102)
GRAY2 = (77, 77, 51)
BLUE = (0, 0, 80)

colors = {0: WHITE, 1: GREEN, -1: GRAY1, -2: GRAY2}

# This sets the WIDTH and HEIGHT of each grid location
WIDTH = 40
HEIGHT = 40

# This sets the margin between each cell
MARGIN = 5

# Initialize pygame
pygame.init()

X_DIM = 20
Y_DIM = 20

# Set the HEIGHT and WIDTH of the screen
WINDOW_SIZE = [(WIDTH + MARGIN) * X_DIM + MARGIN, (HEIGHT + MARGIN) * Y_DIM + MARGIN]
screen = pygame.display.set_mode(WINDOW_SIZE)

# Set title of screen
pygame.display.set_caption("D* Lite Path Planning")

# Loop until the user clicks the close button.
done = False

# Used to manage how fast the screen updates
clock = pygame.time.Clock()

basicfont = pygame.font.SysFont("Comic Sans MS", 36)


def update_gridworlds(occupancy_grid, swarm_state):
    for s in swarm_state:
        s.update_gridworld(occupancy_grid)


def main():

    ### Simulation Constructs ###

    # Create occupancy grid
    occupancy_grid = OccupancyGrid(X_DIM, Y_DIM)

    # Add obstacles
    # occupancy_grid.add_square_obstacle(100, 100, 15)
    # occupancy_grid.add_square_obstacle(175, 150, 35)
    # occupancy_grid.add_square_obstacle(80, 75, 25)
    # occupancy_grid.add_square_obstacle(40, 100, 80)

    visibility_range = 10
    swarm_state = [
        SwarmBotState(
            id=0, pos_x=3, pos_y=3, goal_x=15, goal_y=3, visibility_range=visibility_range
        ),
        SwarmBotState(
            id=1, pos_x=15, pos_y=3, goal_x=3, goal_y=3, visibility_range=visibility_range
        ),
    ]

    # Initialize robot path planning states
    for s in swarm_state:

        # Add navigation goal for visualization
        # occupancy_grid.add_goal(s.goal_x, s.goal_y)

        # Add robots to occupancy grid
        occupancy_grid.add_robot(s.id, s.pos_x, s.pos_y)
    
    for s in swarm_state:

        # Convert occupancy grid to GridWorld
        grid_world_rep = occupancy_grid.convert_to_grid_world_list(s.id)
        # pdb.set_trace()

        # Initialize D* grid representation
        graph = GridWorld(
            occupancy_grid.width, occupancy_grid.height, external_graph=grid_world_rep
        )

        # Add robot start and goal
        s_start = coordsToStateName(s.pos_x, s.pos_y)
        s_goal = coordsToStateName(s.goal_x, s.goal_y)
        graph.setStart(s_start)
        graph.setGoal(s_goal)

        # Add to swarm state
        s.graph = graph

        # Initiliaze D*
        s.graph, s.queue, s.k_m = initDStarLite(s.graph, s.queue, s_start, s_goal, s.k_m)
        s.s_current = s_start

    ### Path Planning Loop ###
    # Meant to reflect the real loop as much as possible

    # Move through grid while dynamically planning path
    while True:
        for s in swarm_state:

            # Update occupancy grid
            # This will be from the CV pipeline, but in the simulation it is not needed
            # Update robot GridWorlds from new occupancy grid

            # Plan the next step for the robot
            s.s_new, s.k_m = moveAndRescan(s.graph, s.queue, s.s_current, s.visibility_range, s.k_m)

            if s.s_new != "goal":

                # Send new position to robots
                # This is where you will send stuff over comms
                # In simulation we simply step forward
                s.s_current = s.s_new

                # Update the obstacle grid
                # This is a simulation construct - the real pipeline will
                # retrieve the latest occupancy grid at the beginning of this loop
                current_x, current_y = stateNameToCoords(s.s_current)
                s.pos_x = current_x
                s.pos_y = current_y
                occupancy_grid.update_robot_position(s.id, current_x, current_y)

                update_gridworlds(occupancy_grid, swarm_state)

        # Set the screen background
        screen.fill(BLACK)

        robot = swarm_state[1]

        # Draw the grid
        for row in range(Y_DIM):
            for column in range(X_DIM):
                color = WHITE
                pygame.draw.rect(
                    screen,
                    colors[robot.graph.cells[row][column]],
                    [
                        (MARGIN + WIDTH) * column + MARGIN,
                        (MARGIN + HEIGHT) * row + MARGIN,
                        WIDTH,
                        HEIGHT,
                    ],
                )
                node_name = "x" + str(column) + "y" + str(row)
                if robot.graph.graph[node_name].g != float("inf"):
                    # text = basicfont.render(
                    # str(graph.graph[node_name].g), True, (0, 0, 200), (255,
                    # 255, 255))
                    text = basicfont.render(str(robot.graph.graph[node_name].g), True, (0, 0, 200))
                    textrect = text.get_rect()
                    textrect.centerx = int(column * (WIDTH + MARGIN) + WIDTH / 2) + MARGIN
                    textrect.centery = int(row * (HEIGHT + MARGIN) + HEIGHT / 2) + MARGIN
                    screen.blit(text, textrect)

        # fill in goal cell with GREEN
        pygame.draw.rect(
            screen,
            GREEN,
            [
                (MARGIN + WIDTH) * robot.goal_x + MARGIN,
                (MARGIN + HEIGHT) * robot.goal_y + MARGIN,
                WIDTH,
                HEIGHT,
            ],
        )
        # draw moving robot, based on pos_coords
        robot_center = [
            int(robot.pos_x * (WIDTH + MARGIN) + WIDTH / 2) + MARGIN,
            int(robot.pos_y * (HEIGHT + MARGIN) + HEIGHT / 2) + MARGIN,
        ]
        pygame.draw.circle(screen, RED, robot_center, int(WIDTH / 2) - 2)

        # draw robot viewing range
        pygame.draw.rect(
            screen,
            BLUE,
            [
                robot_center[0] - visibility_range * (WIDTH + MARGIN),
                robot_center[1] - visibility_range * (HEIGHT + MARGIN),
                2 * visibility_range * (WIDTH + MARGIN),
                2 * visibility_range * (HEIGHT + MARGIN),
            ],
            2,
        )

        # Limit to 60 frames per second
        clock.tick(3)

        # Go ahead and update the screen with what we've drawn.
        pygame.display.flip()

        # Display occupancy grid
        occupancy_grid_bgr = cv2.cvtColor(occupancy_grid.grid, cv2.COLOR_GRAY2BGR)
        occupancy_grid_bgr[occupancy_grid == 0] = [255, 255, 255]
        occupancy_grid_bgr[occupancy_grid == 1] = [0, 0, 0]
        occupancy_grid_bgr[occupancy_grid == 2] = [255, 0, 0]
        occupancy_grid_bgr[occupancy_grid == 3] = [0, 255, 0]

        color_increment = 100
        color = 120
        for s in occupancy_grid.robots:
            cv2.rectangle(
                occupancy_grid_bgr,
                [s.x_min, s.y_min],
                [s.x_max, s.y_max],
                [0, s.id * 100 + 100, 0],
                2,
            )
            color += color_increment

        # for coords in swarm_coordinates:
        # cv2.circle(
        # occupancy_grid_bgr,
        # (int(coords[0]), int(coords[1])),
        # 1,
        # (255, color, 255),
        # -1,
        # )
        # color += color_increment
        cv2.imshow("Occupancy Grid", occupancy_grid_bgr)
        cv2.waitKey(1)

        # time.sleep(0.01)


if __name__ == "__main__":
    main()
