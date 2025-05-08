import numpy as np
from numpy import pi
import random
from copy import deepcopy
import time

from lib.detectCollision import detectCollision
from lib.loadmap import loadmap
from lib.calculateFK import FK

# from detectCollision import detectCollision
# from loadmap import loadmap
# from calculateFK import FK


class Tree:
    def __init__(self):
        self.q = np.ones(7)
        self.parent = None


def check_start_goal(config, obstacles, step_size=0.01):
    is_collision = False
    fk = FK()
    q_start = config - step_size
    q_end = config
    jointPos_start, _ = fk.forward(q_start)
    jointPos_end, _ = fk.forward(q_end)
    for obstacle in obstacles:
        collision = detectCollision(jointPos_start, jointPos_end, obstacle)
        if any(collision):
            is_collision = True
            return is_collision
    return is_collision


def extend(nearest_config, target_config, obstacles, goal_test=False, step_size=0.15):
    is_collision = False
    fk = FK()
    if len(obstacles) == 0:
        q_end = nearest_config + step_size * (target_config - nearest_config)
        return q_end, is_collision

    q_start = nearest_config
    q_end = nearest_config + step_size * (target_config - nearest_config)
    jointPos_start, _ = fk.forward(q_start)
    jointPos_end, _ = fk.forward(q_end)
    for obstacle in obstacles:
        collision = detectCollision(jointPos_start, jointPos_end, obstacle)
        if any(collision):
            is_collision = True
            return None, is_collision
    return q_end, is_collision


def extract_path(tree_nodes):
    i = len(tree_nodes) - 1
    path = [tree_nodes[i].q]
    for _ in range(len(tree_nodes)):
        i = tree_nodes[i].parent
        path.append(tree_nodes[i].q)
        if i == 0:
            break
    path = path[::-1]
    return path


def rrt(map, start, goal):
    """
    Implement RRT algorithm in this file.
    :param map:         the map struct
    :param start:       start pose of the robot (0x7).
    :param goal:        goal pose of the robot (0x7).
    :return:            returns an mx7 matrix, where each row consists of the configuration of the Panda at a point on
                        the path. The first row is start and the last row is goal. If no path is found, PATH is empty
    """

    path = []
    q_start = Tree()
    q_start.q = start
    is_goal = False
    # is_collision = False
    tree_nodes = [q_start]
    lowerLim = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
    upperLim = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

    tolerance = 1.5
    max_iter = 8000
    iter = 0
    obstacles = map.obstacles

    rho_not = 0.15
    pad = [
        -rho_not,
        -rho_not,
        -rho_not,
        rho_not,
        rho_not,
        rho_not,
    ]
    for i in range(len(obstacles)):
        obstacles[i] += pad

    start_collision = check_start_goal(start, obstacles)
    goal_collision = check_start_goal(goal, obstacles)
    if start_collision or goal_collision:
        path = np.array([])
        return path

    while not is_goal and iter <= max_iter:
        # Find a random configuration
        q_random = np.random.uniform(lowerLim, upperLim)

        # Find in Tree a configuration nearest to the random configuration
        nodes_list = [node.q for node in tree_nodes]
        nearest_config_idx = np.argmin(np.linalg.norm(nodes_list - q_random, axis=1))
        nearest_config = nodes_list[nearest_config_idx]

        # Extend the tree
        q_new, is_collision = extend(nearest_config, q_random, obstacles)
        iter += 1

        if is_collision:
            continue

        newNode = Tree()
        newNode.q = q_new
        newNode.parent = nearest_config_idx
        tree_nodes.append(newNode)

        if np.linalg.norm(q_new - goal) < tolerance:
            # is_collision = False
            # print("Testing possible goal...")
            _, is_collision = extend(q_new, goal, obstacles, goal_test=True)

            if is_collision == False:
                last_node_idx = len(tree_nodes) - 1
                goalNode = Tree()
                goalNode.q = goal
                goalNode.parent = last_node_idx
                tree_nodes.append(goalNode)
                is_goal = True
                # print("Goal is reached --> Path is found!")

        # if iter % 500 == 0:
        #     nodes_list = [node.q for node in tree_nodes]
        #     errors = np.linalg.norm(nodes_list - goal, axis=1)
        #     nearest_config_idx = np.argmin(errors)
        #     nearest_config = nodes_list[nearest_config_idx]
        #     print("Iteration {}".format(iter))
        #     print("\tError = ", errors[nearest_config_idx])
        #     print("\tConfiguration closest to Goal = \n\t{}".format(nearest_config))

    print("Number of iterations: ", iter)
    if is_goal == True:
        path = extract_path(tree_nodes=tree_nodes)
    else:
        path = np.array([])
    return path


if __name__ == "__main__":
    map_struct = loadmap("maps/map5.txt")
    
    # # MAP 1
    # start = np.array([0, -1, 0, -2, 0, 1.57, 0])
    # goal = np.array([-1.2, 1.57, 1.57, -2.07, -1.57, 1.57, 0.7])
    
    # # MAP 2
    # start = np.array([0, -1, 0, -2.5, 0, 2.7, 0.707])
    # goal = np.array([1.9, 1.57, -1.57, -1.57, 1.57, 1.57, 0.707])
    
    # # MAP 3
    # start = np.array([-1.57, -1, 0, -2.5, 0, 2.7, 0.707])
    # goal = np.array([1.9, 1.57, -1.57, -1.57, 1.57, 1.57, 0.707])
    
    # # MAP 4
    # start = np.array([-1.57, 1.57, 1.57, -1.57, 0, 2.7, 0.707])
    # goal = np.array([1.9, 1.57, -1.57, -1.57, 0, 1.57, 0.707])
    
    # MAP 5
    start = np.array([ -pi/4,    0,     0, -pi/2,     0, pi/2, pi/4 ])
    goal = np.array([ pi/4,    0,     0, -pi/2,     0, pi/2, pi/4 ])
    
    start_time = time.time()
    q_path = rrt(deepcopy(map_struct), deepcopy(start), deepcopy(goal))
    end_time = time.time()
    print("Duration: {}".format(end_time - start_time))

    if len(q_path) == 0:
        print("No path was found.")
    else:
        print("Number of waypoints: {}".format(len(q_path)))
        for i in range(len(q_path)):
            if i < 3 or i > len(q_path) - 4:
                print(f"Waypoint {i}", "\n\tq =", q_path[i])
        # print("q path: ", q_path)
