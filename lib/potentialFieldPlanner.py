import numpy as np
from math import pi, acos

# from scipy.linalg import null_space
from copy import deepcopy
import random

from lib.calcJacobian import calcJacobian
from lib.calculateFK import FK
from lib.detectCollision import detectCollision
from lib.loadmap import loadmap

# from calcJacobian import calcJacobian
# from calculateFK import FK
# from detectCollision import detectCollision
# from loadmap import loadmap

# import matplotlib.pyplot as plt


class PotentialFieldPlanner:
    # JOINT LIMITS
    lower = np.array([-2.8973, -1.7628, -2.8973, -
                     3.0718, -2.8973, -0.0175, -2.8973])
    upper = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

    center = (
        lower + (upper - lower) / 2
    )
    # compute middle of range of motion of each joint
    fk = FK()

    # potential field controller parameters
    att_const = 5  # attracting force constant
    rep_const = 0.1  # repulsive force constant
    rho_not = 0.2  # distance of influence of obstacle
    alpha = 0.02  # learning rate

    def __init__(self, tol=1e-4, max_steps=2000, min_step_size=1e-5):
        """
        Constructs a potential field planner with solver parameters.

        PARAMETERS:
        tol - the maximum distance between two joint sets
        max_steps - number of iterations before the algorithm must terminate
        min_step_size - the minimum step size before concluding that the
        optimizer has converged
        """

        # YOU MAY NEED TO CHANGE THESE PARAMETERS

        # solver parameters
        self.tol = tol
        self.max_steps = max_steps
        self.min_step_size = min_step_size

    ######################
    ## Helper Functions ##
    ######################
    # The following functions are provided to you to help you to better structure your code
    # You don't necessarily have to use them. You can also edit them to fit your own situation

    @staticmethod
    def attractive_force(target, current):
        """
        Helper function for computing the attactive force between the current position and
        the target position for one joint. Computes the attractive force vector between the
        target joint position and the current joint position

        INPUTS:
        target - 3x1 numpy array representing the desired joint position in the world frame
        current - 3x1 numpy array representing the current joint position in the world frame

        OUTPUTS:
        att_f - 3x1 numpy array representing the force vector that pulls the joint
        from the current position to the target position
        """

        # att_f = np.zeros((3, 1))
        # d = 0.5  # transition distance between conic to parabolic well potential
        # zeta = PotentialFieldPlanner.att_const
        # diff = current - target
        # norm = np.linalg.norm(diff)

        # # If distance is less than or equal to threshold, use PARABOLIC
        # if norm <= d:
        #     att_f = -zeta * diff

        # # If distance is greater than threshold, use CONIC
        # else:
        #     att_f = -d * zeta * diff / norm

        # return att_f  # .flatten()
        
        zeta = PotentialFieldPlanner.att_const
        target = target.reshape(3, 1)
        current = current.reshape(3, 1)
        assert target.shape == (3, 1) and current.shape == (3, 1)
        att_f = - (current - target)
        assert att_f.shape == (3, 1)

        # END STUDENT CODE

        return att_f * zeta

    @staticmethod
    def repulsive_force(obstacle, current, unitvec=np.zeros((3, 1))):
        """
        Helper function for computing the repulsive force between the current position
        of one joint and one obstacle. Computes the repulsive force vector between the
        obstacle and the current joint position

        INPUTS:
        obstacle - 1x6 numpy array representing the an obstacle box in the world frame
        current - 3x1 numpy array representing the current joint position in the world frame
        unitvec - 3x1 numpy array representing the unit vector from the current joint position
        to the closest point on the obstacle box

        OUTPUTS:
        rep_f - 3x1 numpy array representing the force vector that pushes the joint
        from the obstacle
        """

        # print("Obstacle in the loop = {}".format(obstacle))

        # rep_f = np.zeros((3, 1))
        # rho_not = PotentialFieldPlanner.rho_not
        # eta = PotentialFieldPlanner.rep_const
        # obstacle = np.transpose(obstacle)
        # current = np.reshape(current, (1, len(current)))

        # rho_i, unitvec = PotentialFieldPlanner.dist_point2box(
        #     current, obstacle)
        # rho_i = rho_i[0]
        # unitvec = unitvec.reshape((3, 1))
        # # print("unitvec = {}".format(unitvec))
        # # print("Shape of unitvec = {}".format(unitvec.shape))
        # # print("rho_i = {}".format(rho_i))
        # # print("rho_not = {}".format(rho_not))
        # # print("Obstacle = {}".format(obstacle))

        # # If distance is too close to obstacle, repulsive force equals infinity
        # if rho_i <= 0.01:
        #     rep_f = np.array([np.inf, np.inf, np.inf]).reshape((3, 1))

        # # If distance is smaller than rho_not, force activates
        # elif 0.01 < rho_i <= rho_not:
        #     rep_f = eta * (1 / rho_i - 1 / rho_not) * (1 / rho_i**2) * unitvec
        #     # print("rep_f = {}".format(rep_f))
        #     # print("Shape of rep_f = {}".format(rep_f.shape))

        # # If distance is far from obstacle, repulsive force equals 0
        # else:
        #     rep_f = np.zeros((3, 1))
        # return rep_f  # .flatten()
        # # TODO: CHECK THE SIGN OF REP_F

        rho_0 = PotentialFieldPlanner.rho_not
        eta = PotentialFieldPlanner.rep_const
        current = current.reshape(3, 1)
        assert current.shape == (3, 1) and unitvec.shape == (3, 1)

        rep_f = np.zeros((3, 1))
        dist, unit = PotentialFieldPlanner.dist_point2box(current.T, obstacle)
        assert dist.shape == (1,) and unit.shape == (1, 3)

        if dist > rho_0:
            pass
        elif dist == 0:
            rep_f = 1e6 * (-unitvec.T)
            # assert False, "The current joint position is inside the obstacle"
        else:
            rep_f = (1/dist - 1/rho_0) * (1/dist**2) * (-unit.T)
        rep_f = rep_f.reshape(3, 1)
        assert rep_f.shape == (3, 1)

        return rep_f * eta

    @staticmethod
    def dist_point2box(p, box):
        """
        Helper function for the computation of repulsive forces. Computes the closest point
        on the box to a given point

        INPUTS:
        p - nx3 numpy array of points [x,y,z]
        box - 1x6 numpy array of minimum and maximum points of box

        OUTPUTS:
        dist - nx1 numpy array of distance between the points and the box
                dist > 0 point outside
                dist = 0 point is on or inside box
        unit - nx3 numpy array where each row is the corresponding unit vector
        from the point to the closest spot on the box
            norm(unit) = 1 point is outside the box
            norm(unit)= 0 point is on/inside the box

         Method from MultiRRomero
         @ https://stackoverflow.com/questions/5254838/
         calculating-distance-between-a-point-and-a-rectangular-box-nearest-point
        """
        # THIS FUNCTION HAS BEEN FULLY IMPLEMENTED FOR YOU

        # Get box info
        boxMin = np.array([box[0], box[1], box[2]])
        boxMax = np.array([box[3], box[4], box[5]])
        boxCenter = boxMin * 0.5 + boxMax * 0.5
        p = np.array(p)

        # Get distance info from point to box boundary
        dx = np.amax(
            np.vstack(
                [boxMin[0] - p[:, 0], p[:, 0] -
                    boxMax[0], np.zeros(p[:, 0].shape)]
            ).T,
            1,
        )
        dy = np.amax(
            np.vstack(
                [boxMin[1] - p[:, 1], p[:, 1] -
                    boxMax[1], np.zeros(p[:, 1].shape)]
            ).T,
            1,
        )
        dz = np.amax(
            np.vstack(
                [boxMin[2] - p[:, 2], p[:, 2] -
                    boxMax[2], np.zeros(p[:, 2].shape)]
            ).T,
            1,
        )

        # convert to distance
        distances = np.vstack([dx, dy, dz]).T
        dist = np.linalg.norm(distances, axis=1)

        # Figure out the signs
        signs = np.sign(boxCenter - p)

        # Calculate unit vector and replace with
        unit = distances / dist[:, np.newaxis] * signs
        unit[np.isnan(unit)] = 0
        unit[np.isinf(unit)] = 0
        return dist, unit

    @staticmethod
    def compute_forces(target, obstacle, current):
        """
        Helper function for the computation of forces on every joints. Computes the sum
        of forces (attractive, repulsive) on each joint.

        INPUTS:
        target - 3x7 numpy array representing the desired joint/end effector positions
        in the world frame
        obstacle - nx6 numpy array representing the obstacle box min and max positions
        in the world frame
        current- 3x7 numpy array representing the current joint/end effector positions
        in the world frame

        OUTPUTS:
        joint_forces - 3x7 numpy array representing the force vectors on each
        joint/end effector
        """

        joint_forces = np.zeros((3, 7))
        # F_att = np.zeros((3, 7))
        # F_rep = np.zeros((3, 7))

        # # Calculate attractive forces
        # for i in range(7):
        #     F_att[:, i] = PotentialFieldPlanner.attractive_force(
        #         target[:, i], current[:, i]
        #     ).flatten()
        # # print("Attractive forces = {}".format(F_att))

        # # print("Obstacle = \n{}".format(obstacle))
        # # print("Obstacle shape = \n{}".format(obstacle.shape))
        # # print("Number of obstacles = \n{}".format(len(obstacle)))
        # # print("First Obstacle = \n{}".format(obstacle[0]))

        # # print("Current configuration = \n{}".format(current))
        # # print("Origin of the first configuration = \n{}".format(current[:, 0]))

        # # Calculate repulsive forces
        # for j in range(7):
        #     for k in range(len(obstacle)):
        #         F_rep[:, j] += PotentialFieldPlanner.repulsive_force(
        #             obstacle[k], current[:, j]
        #         ).flatten()
        # # print("Repulsive forces = {}".format(F_rep))

        # # Sum attractive and repulsive forces
        # joint_forces = F_att + F_rep  # 3x7

        for i in range(7):
            att_force = PotentialFieldPlanner.attractive_force(
                target[:, i], current[:, i])
            joint_forces[:, i] += att_force.reshape(3,)
            if len(obstacle) > 0:
                for j in range(obstacle.shape[0]):
                    rep_f = PotentialFieldPlanner.repulsive_force(
                        obstacle[j, :], current[:, i])
                    joint_forces[:, i] += rep_f.reshape(3,)

        return joint_forces

    @staticmethod
    def compute_torques(joint_forces, q):
        """
        Helper function for converting joint forces to joint torques. Computes the sum
        of torques on each joint.

        INPUTS:
        joint_forces - 3x7 numpy array representing the force vectors on each
        joint/end effector
        q - 1x7 numpy array representing the current joint angles

        OUTPUTS:
        joint_torques - 1x7 numpy array representing the torques on each joint
        """
        # joint_torques = np.zeros((1, 7))

        # Jv = calcJacobian(q)[:3, :]  # 3x7
        # Jv_i = np.zeros((3, 7))
        # for i in range(7):
        #     Jv_i[:, i] = Jv[:, i]  # 3x7
        #     joint_torques += (Jv_i.T @ joint_forces[:, i]).reshape((1, 7))

        J = calcJacobian(q)
        Jv = J[:3, :]
        assert Jv.shape == (3, 7) and joint_forces.shape == (3, 7)

        joint_torques = np.zeros((1, 7))

        for i in range(7):
            torque = Jv[:, :i+1].T @ joint_forces[:, i]
            assert torque.shape == (i+1,)
            joint_torques[:, :i+1] += torque

        return joint_torques

    @staticmethod
    def q_distance(target, current):
        """
        Helper function which computes the distance between any two
        vectors.

        This data can be used to decide whether two joint sets can be
        considered equal within a certain tolerance.

        INPUTS:
        target - 1x7 numpy array representing some joint angles
        current - 1x7 numpy array representing some joint angles

        OUTPUTS:
        distance - the distance between the target and the current joint sets

        """

        distance = np.linalg.norm(target - current)
        # print("distance = {}".format(distance))
        return distance

    @staticmethod
    def compute_gradient(q, target, map_struct):
        """
        Computes the joint gradient step to move the current joint positions to the
        next set of joint positions which leads to a closer configuration to the goal
        configuration

        INPUTS:
        q - 1x7 numpy array. the current joint configuration, a "best guess" so far for the final answer
        target - 1x7 numpy array containing the desired joint angles
        map_struct - a map struct containing the obstacle box min and max positions

        OUTPUTS:
        dq - 1x7 numpy array. a desired joint velocity to perform this task
        """

        # dq = np.zeros((1, 7))
        # fk = FK()

        # targetPos, _ = fk.forward(target)
        # currentPos, _ = fk.forward(q)

        # # print("Target position = \n{}".format(targetPos[1::].T))
        # # print("Current position = \n{}".format(currentPos[1::].T))
        # # print("Obstacle map = \n{}".format(map_struct))

        # joint_forces = PotentialFieldPlanner.compute_forces(
        #     targetPos[1::].T,
        #     map_struct,
        #     currentPos[1::].T,
        # )

        # joint_torques = PotentialFieldPlanner.compute_torques(
        #     joint_forces, q)  # 1X7

        # if np.linalg.norm(joint_torques) != 0:
        #     dq = joint_torques / np.linalg.norm(joint_torques)  # 1X7

        obstacle = map_struct.obstacles

        q = q.reshape(7,)
        target = target.reshape(7,)
        assert q.shape == (7,) and target.shape == (7,)

        fk = FK()
        target_joints, _ = fk.forward(target)
        target_joints = (target_joints[1:, :]).T

        current_joints, _ = fk.forward(q)
        current_joints = (current_joints[1:, :]).T
        assert target_joints.shape == (3, 7) and current_joints.shape == (3, 7)

        joint_forces = PotentialFieldPlanner.compute_forces(
            target_joints, obstacle, current_joints)
        # print('joint_forces', joint_forces)

        joint_torques = PotentialFieldPlanner.compute_torques(joint_forces, q)
        # print('joint_torques', joint_torques)

        dq = joint_torques

        return dq

    ###############################
    ### Potential Feild Solver  ###
    ###############################

    def plan(self, map_struct, start, goal):
        """
        Uses potential field to move the Panda robot arm from the starting configuration to
        the goal configuration.

        INPUTS:
        map_struct - a map struct containing min and max positions of obstacle boxes
        start - 1x7 numpy array representing the starting joint angles for a configuration
        goal - 1x7 numpy array representing the desired joint angles for a configuration

        OUTPUTS:
        q - nx7 numpy array of joint angles [q0, q1, q2, q3, q4, q5, q6]. This should contain
        all the joint angles throughout the path of the planner. The first row of q should be
        the starting joint angles and the last row of q should be the goal joint angles.
        """

        # q_path = np.array([]).reshape(0, 7)
        # fk = FK()
        # q_curr = start
        # # q_path = np.vstack((q_path, [start]))
        # steps = 0
        # errors = []
        # initial_lr = PotentialFieldPlanner.alpha
        # decay_factor = 0.8
        # epoch = 0

        # map_struct_1 = map_struct.obstacles

        # while steps < self.max_steps:
        #     # The following comments are hints to help you to implement the planner
        #     # You don't necessarily have to follow these steps to complete your code

        #     # Compute gradient
        #     # TODO: this is how to change your joint angles
        #     dq = PotentialFieldPlanner.compute_gradient(
        #         q_curr, goal, map_struct_1
        #     ).flatten()
        #     error = PotentialFieldPlanner.q_distance(goal, q_curr)
        #     q_path = np.vstack((q_path, q_curr))

        #     # Termination Conditions
        #     # TODO: check termination conditions
        #     if error < self.tol:
        #         q_path = np.vstack([q_path, [goal]])
        #         return q_path

        #     # YOU NEED TO CHECK FOR COLLISIONS WITH OBSTACLES
        #     # TODO: Figure out how to use the provided function
        #     jointPos, _ = fk.forward(q_curr)
        #     line_pt_1 = jointPos[:7, :]
        #     line_pt_2 = jointPos[1:, :]

        #     for i in range(len(map_struct_1)):
        #         if any(detectCollision(line_pt_1, line_pt_2, map_struct_1[i])):
        #             # print("Collision detected!")
        #             steps = steps - 3
        #             q_path = q_path[:-3]
        #             q_curr = q_path[-1]
        #             dq = np.random.randn(1, 7).flatten()*2

        #     # YOU MAY NEED TO DEAL WITH LOCAL MINIMA HERE
        #     # TODO: when detect a local minima, implement a random walk
        #     if steps >= 4:
        #         diff_1 = np.linalg.norm(q_path[steps - 1] - q_path[steps])
        #         diff_2 = np.linalg.norm(q_path[steps - 2] - q_path[steps])
        #         diff_3 = np.linalg.norm(q_path[steps - 3] - q_path[steps])
        #         diff_4 = np.linalg.norm(q_path[steps - 4] - q_path[steps])

        #         if (
        #             diff_1 < self.min_step_size
        #             and diff_2 < self.min_step_size
        #             and diff_3 < self.min_step_size
        #             and diff_4 < self.min_step_size
        #         ):
        #             dq = np.random.randn(1, 7).flatten()*2
        #             print("Local minima detected at Iter {}!".format(steps))

        #     if steps % 50 == 0:
        #         lr = initial_lr * (decay_factor**epoch)
        #         epoch += 1

        #     # errors.append(PotentialFieldPlanner.q_distance(goal, q_curr))
        #     q_curr = q_curr + lr * dq
        #     steps += 1

        #     # print(
        #     #     "Iter {}: \n\t q_step = {}\n\t error: {}".format(
        #     #         steps, q_path[steps], error
        #     #     )
        #     # )

        # # plt.plot(np.array(errors))
        # # plt.show()

        alpha = PotentialFieldPlanner.alpha
        q_path = np.array([]).reshape(0, 7)

        q = start.reshape(1, 7)
        goal = goal.reshape(1, 7)
        i = 0

        while True:
            q_path = np.vstack((q_path, q))
            i += 1

            # Compute gradient
            # TODO: this is how to change your joint angles
            dq = PotentialFieldPlanner.compute_gradient(q, goal, map_struct)

            # Termination Conditions
            # TODO: check termination conditions
            if i > self.max_steps or self.q_distance(goal[:, :6], q[:, :6]) < self.tol:
                break  # exit the while loop if conditions are met!

            # YOU NEED TO CHECK FOR COLLISIONS WITH OBSTACLES
            # TODO: Figure out how to use the provided function

            # YOU MAY NEED TO DEAL WITH LOCAL MINIMA HERE
            # TODO: when detect a local minima, implement a random walk
            if np.linalg.norm(dq) < self.min_step_size:
                dq = np.random.randn(1, 7)*5
                assert False, 'random walk not implemented'

            dq = dq / (np.linalg.norm(dq) + 1e-16)

            # dq[:, -1] += 0.1 * (goal[:, -1] - q[:, -1])

            # Update q
            assert dq.shape == q.shape
            q = q + alpha * dq

        return q_path


################################
## Simple Testing Environment ##
################################

if __name__ == "__main__":
    np.set_printoptions(suppress=True, precision=5)

    planner = PotentialFieldPlanner()

    # inputs
    # map_struct = loadmap("maps/map1.txt")
    # start = np.array([0, -1, 0, -2, 0, 1.57, 0])
    # goal = np.array([-1.2, 1.57, 1.57, -2.07, -1.57, 1.57, 0.7])
    
    
    map_struct = loadmap("maps/map5.txt")
    start = np.array([ -pi/4,    0,     0, -pi/2,     0, pi/2, pi/4 ])
    goal = np.array([ pi/4,    0,     0, -pi/2,     0, pi/2, pi/4 ])

    # potential field planning
    q_path = planner.plan(deepcopy(map_struct),
                          deepcopy(start), deepcopy(goal))

    # show results
    for i in range(q_path.shape[0]):
        error = PotentialFieldPlanner.q_distance(q_path[i, :], goal)
        print(
            "iteration:", i, " q =", q_path[i, :], " error={error}".format(
                error=error)
        )

    print("q path: ", q_path)
