import sys
import operator
from math import pi, sin, cos
import numpy as np
from time import perf_counter, sleep
import copy
from scipy.spatial.transform import Rotation as Rot

import rospy

# Common interfaces for interacting with both the simulation and real environments!
from core.interfaces import ArmController
from core.interfaces import ObjectDetector

# for timing that is consistent with simulation or real time as appropriate
from core.utils import time_in_seconds

# Import helper files from our code solutions
from lib.calculateFK import FK
from lib.IK_position_null import IK


class MAIN_ROUTINE:
    def __init__(self, robot_params):
        """
        Initialization for routines (for both dynamic and static stacking)
        :param robot_params: list of params depending on RED/BLUE team
        """
        self.H_ee_camera = detector.get_H_ee_camera()
        # IN WORLD FRAME, the rest in robot frame
        self.robot_center = robot_params[0]
        self.dynamic_table_center = robot_params[1]

        # Joint configs for known positions:
        self.goal_config = robot_params[2]
        self.start_static_config = robot_params[3]
        self.start_dynamic_config = robot_params[4]

        # Gripper offsets, depending on robot used
        self.grab_offset_x = robot_params[5]
        self.grab_offset_y = robot_params[6]

        # Predefined configs for movements
        self.pre_stack_config = robot_params[7]
        self.stack_config = robot_params[8]
        self.move_clear_config = robot_params[9]

        # Predefined configs for dynamic (wait and grab) routine
        self.dy_pre_grab = robot_params[10]
        self.dy_pre_grab_2 = robot_params[11]
        self.dy_grab = robot_params[12]
        self.dy_post_grab = robot_params[13]
        self.dy_pre_stack = robot_params[14]
        self.dy_stack = robot_params[15]
        self.dy_clear = robot_params[16]
        self.dy_grab_HeeW = robot_params[17]

        self.start_static_config_alt = robot_params[18]
        self.dy_post_stack = robot_params[19]

        # Initialize/set number of blocks
        self.num_static_blocks = 4
        self.num_stacked_blocks = 0
        self.num_dynamic_blocks = 8
        self.num_stacked_dyn_blocks = 0
        self.max_stack = 8

        # Turn table angular speed
        # self.omega = 0.02248
        # self.omega = 0.0642
        # self.omega = 0.095
        self.omega = 0.1

        # Angle rotation
        # self.d_theta = 0.17453  # Approx 10 degrees
        # self.d_theta = 0.34907  # Approx 20 degrees
        # self.d_theta = 0.5236  # Approx 30 degrees
        self.d_theta = 0.873  # Approx 50 degrees
        # self.d_theta = pi/3  # 60 degrees

        # Anticipation time before grasping
        self.anticipation_time = 1

        # Max distance from end effector for block selection
        self.max_dist = 0.40
        self.min_dist = 0.15


    def go_static(self):
        """
        Main STATIC block stacking routine
        :return: num_stacked_blocks: number of successfully stacked static blocks
        """

        is_static = True
        while is_static and self.num_stacked_blocks < self.num_static_blocks:
            # 1) MOVE TO START POSITION FOR STATIC ROUTINE
            print("MOVING TO START STATIC POSITION...")
            arm.safe_move_to_position(self.start_static_config)
            arm.open_gripper()

            # 2) DETECT AND COUNT STATIC BLOCKS
            print("DETECTING STATIC BLOCKS...")
            block_pose_list = detector.get_detections()

            # Compare number of blocks against block counter
            num_det_blocks = len(block_pose_list)
            if num_det_blocks == 0:
                print("NO BLOCKS DETECTED, ALTERNATE VIEW ANGLE...")
                arm.safe_move_to_position(self.start_static_config_alt)
                block_pose_list = detector.get_detections()

                num_det_blocks = len(block_pose_list)
                if num_det_blocks == 0:
                    print("NO BLOCKS DETECTED AGAIN, EXITING STATIC ROUTINE...")
                    return self.num_stacked_blocks
            
            # 3) CHOOSE BLOCK TO PICK
            # Function to choose next block to pick.
            H_o_w = self.select_block(block_pose_list)
            print("SELECTED A BLOCK, GOING FOR IT...")

            # 4) GRAB AND PLACE BLOCK
            grab_place_status = self.grab_and_place(H_o_w, self.num_stacked_blocks)

            if grab_place_status:
                self.num_stacked_blocks += 1
                print(
                    "GRAB AND PLACE SUCCESS! Stacked blocks: {}".format(
                        self.num_stacked_blocks
                    )
                )
            else:
                print("GRAB FAILURE!")

        arm.safe_move_to_position(self.move_clear_config[-1])

    def go_dynamic_wait_grab(self):
        """
        DYNAMIC block stacking routine using wait and grab method
        """

        while self.num_stacked_blocks < self.max_stack:
            # 1) Move to start
            print("MOVING TO START DYNAMIC POSITION...")
            arm.open_gripper()
            arm.safe_move_to_position(self.dy_pre_grab)
            arm.safe_move_to_position(self.dy_pre_grab_2)

            # 2) Move to grab position, wait, and grab. Try until get something
            grabbed = False
            while not grabbed:
                arm.safe_move_to_position(self.dy_grab)
                print("WAITING...")
                sleep(3)  # wait for 5 seconds for block

                # Alternate routine to check nearest block #TODO
                # print("WAITING for Nearest block...")
                # buffer_time = 0.0  # add 1 seconds buffer
                # wait_time = self.time_to_wait()
                # sleep(wait_time + buffer_time)

                # Closing gripper and checking if successfully grabbed
                print("CLOSING GRIPPER...")
                arm.exec_gripper_cmd(0.048, 50)
                grasp_status = arm.get_gripper_state()
                grasp_gap = (
                    grasp_status.get("position")[0] + grasp_status.get("position")[1]
                )
                print("Grasp status", grasp_status)
                print("Grasp gap", grasp_gap)

                if 0.03 < grasp_gap < 0.055:
                    print("Got something!")
                    grabbed = True
                elif grasp_gap < 0.02:
                    print("Grabbed nothing...trying again...")
                    arm.open_gripper()

            # 3) Go stack
            arm.safe_move_to_position(self.dy_post_grab)
            arm.safe_move_to_position(self.dy_pre_stack[-1])
            arm.safe_move_to_position(self.dy_stack[self.num_stacked_blocks])
            arm.open_gripper()
            arm.safe_move_to_position(self.dy_post_stack)

            # 4) Add to count
            self.num_stacked_blocks += 1
            print(
                "DYNAMIC BLOCK SUCCESS! Stacked blocks: {}".format(
                    self.num_stacked_blocks
                )
            )

    def time_to_wait(self):
        pose_list = detector.get_detections()

        # Find which the nearest block
        best_block = 0
        best_block_dist = -np.inf
        for i, pose in enumerate(pose_list):
            if np.linalg.norm(pose[1][:3, 3]) > best_block_dist:
                best_block = i
                best_block_dist = np.linalg.norm(pose[1][:3, 3])

        print("Block distance:", best_block_dist)
        # Compute target object coordinate (block) relative to the world frame
        H_o_c = pose_list[best_block][1]
        H_c_ee = copy.deepcopy(self.H_ee_camera)
        H_ee_w = copy.deepcopy(self.dy_grab_HeeW)
        H_o_w = H_ee_w @ H_c_ee @ H_o_c

        # Compute time to wait
        block_xy = copy.deepcopy(H_o_w[0:3, 3]) - copy.deepcopy(
            self.dynamic_table_center[0:3]
        )
        block_theta = np.arctan(abs(block_xy[0] / block_xy[1]))

        wait_time = block_theta / self.omega
        print("Wait time: ", wait_time)

        return wait_time

    def go_dynamic_traj(self):
        """
        Performs the grab dynamic block routine
        """
        print("MOVING TO START DYNAMIC POSITION...")
        arm.safe_move_to_position(self.start_dynamic_config)

        while self.num_stacked_dyn_blocks < self.num_dynamic_blocks:
            # Open gripper.
            arm.open_gripper()

            # Detect blocks, get grasping pose and time to grasp
            block_id, target_block_xyz, t_grasp = self.get_block_trajectory()

            # Grab and place block
            print("SELECTED A BLOCK, GOING FOR IT...")
            grab_place_success = self.grab_and_place(
                target_block_xyz, self.num_stacked_blocks, is_static=False, tf=t_grasp
            )

            if grab_place_success:
                self.num_stacked_blocks += 1
                self.num_stacked_dyn_blocks += 1
                print(
                    "GRAB AND PLACE SUCCESS! Stacked blocks: {}".format(
                        self.num_stacked_blocks
                    )
                )
            else:
                print("GRAB FAILURE!")

            # Loop back to start position
            arm.safe_move_to_position(self.start_dynamic_config)

        print("Finished picking dynamic blocks")

        return

    ##############################
    ##  BLOCK DETECTION HELPERS ##
    ##############################

    def select_block(self, block_pose_list):
        H_o_c = block_pose_list[0][1]
        H_c_ee = copy.deepcopy(self.H_ee_camera)

        # Get q of current arm position to get H_ee_W.
        q = arm.get_positions()
        _, H_ee_w = fk.forward(q)

        # Compute target object coordinate (block) relative to the world frame.
        H_o_w = H_ee_w @ H_c_ee @ H_o_c

        # Compute block orientation and position relative to world frame.
        block_ori = correct_R(copy.deepcopy(H_o_w[0:3, 0:3]))
        block_pos = copy.deepcopy(H_o_w[0:3, 3]) + np.array(
            [self.grab_offset_x, self.grab_offset_y, 0]
        )
        # print(f'block_ori: {block_ori}')
        # print(f'block_pos: {block_pos}')
        updated_H_o_w = np.hstack(
            [block_ori.reshape((3, 3)), block_pos.reshape((3, 1))]
        )
        updated_H_o_w = np.vstack([updated_H_o_w, np.array([0, 0, 0, 1])])

        return updated_H_o_w

    def get_ik_sol(
        self,
        target_config_xyz,
        seed=np.array([0.0, 0.0, 0.0, -1.5708, 0.0, 1.8675, 0.0]),
    ):
        """
        Creates a valid IK solution

        INPUTS:
        target_config_xyz - 4x4 np array, transformation matrix for target end-effector position
        seed - starting seed for solver (default seed is neutral position)

        OUTPUTS:
        position_q - 1x7 numpy array, joint positions for use in arm.safe_move_to_position
        ik_success
        """
        position_q, _, ik_success, _ = ik.inverse(
            target_config_xyz, seed, method="J_pseudo", alpha=0.5
        )

        if not ik_success:
            print("Couldn't find solution for target!")

        # TODO: if ik_success still false??
        return position_q, ik_success

    def grab_and_place(
        self, target_block_xyz, num_stacked_blocks, is_static=True, tf=None
    ):
        grab_place_success = True

        if is_static:
            ##############################
            # MOVE TO PRE GRASP POSITION #
            ##############################
            pre_grasp_xyz = copy.deepcopy(target_block_xyz)
            # Height of table + 2.5cm offset above top of grasping block
            pre_grasp_xyz[2, 3] = 0.2 + 0.075
            seed = self.start_static_config
            pre_grasp_q, ik_success = self.get_ik_sol(pre_grasp_xyz, seed)
            if ik_success:
                arm.safe_move_to_position(pre_grasp_q)
            else:
                print("Can't move to pre-grasp position")
                return not grab_place_success

            ##############################
            ### MOVE TO GRASP POSITION ###
            ##############################
            grasp_xyz = copy.deepcopy(target_block_xyz)
            # Hardcode z-axis to be center of block (table + 2.5cm)
            grasp_xyz[2, 3] = 0.2 + 0.025
            grasp_q, ik_success = self.get_ik_sol(grasp_xyz, pre_grasp_q)

            arm.safe_move_to_position(grasp_q)

            # Closing gripper and checking if successful04ly grabbed
            print("CLOSING GRIPPER...")
            arm.exec_gripper_cmd(0.048, 50)
            grasp_status = arm.get_gripper_state()
            grasp_gap = (
                grasp_status.get("position")[0] + grasp_status.get("position")[1]
            )
            # print("grasp position: ", grasp_status.get("position"))
            if 0.045 < grasp_gap < 0.055:
                print("Successfully grasped at ", grasp_gap)
            elif grasp_gap < 0.02:
                print(
                    "Gripper failed to grab the block!\n\t Moving to starting position . . ."
                )
                return not grab_place_success

            self.move_clear_of_blocks(grasp_xyz)

        elif not is_static:
            ##############################
            ### MOVE TO GRASP POSITION ###
            ##############################
            grasp_xyz = copy.deepcopy(target_block_xyz)
            # Hardcode z-axis to be center of block (table + 2.5cm)
            grasp_xyz[2, 3] = 0.2 + 0.045  # TODO: change to 0.025 later
            grasp_q, ik_success = self.get_ik_sol(grasp_xyz, self.start_dynamic_config)

            if not ik_success:
                print("Can't move to pre-grasp position")
                return not grab_place_success

            # Grasping time already passed, return fail grab
            if time_in_seconds() > (tf - self.anticipation_time):
                print("time to grasp =", tf)
                print("time now =", time_in_seconds())
                print("Time to grasp already passed. Did not grasp block!")
                return not grab_place_success

            # Grasping time not yet passed
            else:
                # Wait for grasping time
                print_wait_flag = True
                while time_in_seconds() < tf:
                    if print_wait_flag:
                        print("Waiting to grasp...")
                        print_wait_flag = False

                # Move to target block.
                print("time to grasp =", tf)
                print("time now =", time_in_seconds())
                print("Grasping now...")
                arm.safe_move_to_position(grasp_q)

                # Close gripper.
                print("CLOSING GRIPPER...")

                arm.exec_gripper_cmd(0.048, 50)  # original 0.03

                # Check if gripper has successfully grabbed block.
                grasp_status = arm.get_gripper_state()
                grasp_gap = (
                    grasp_status.get("position")[0] + grasp_status.get("position")[1]
                )
                print("grasp position: ", grasp_status.get("position"))
                if 0.045 < grasp_gap < 0.055:
                    print("Successfully grasped at ", grasp_gap)
                elif grasp_gap < 0.02:
                    print(
                        "Gripper failed to grab the block!\n\t Moving to starting position . . ."
                    )
                    return not grab_place_success

                self.move_clear_of_blocks(grasp_xyz)

        # Move to pre stack
        pre_stack_q = self.pre_stack_config[num_stacked_blocks]
        arm.safe_move_to_position(pre_stack_q)

        # Move to stack
        stack_q = self.stack_config[num_stacked_blocks]
        arm.safe_move_to_position(stack_q)

        # Open gripper
        arm.open_gripper()

        # Move clear
        move_clear_q = self.move_clear_config[num_stacked_blocks]
        arm.safe_move_to_position(move_clear_q)

        return grab_place_success

    def move_clear_of_blocks(self, clear_stacks_xyz):
        clear_stacks_xyz[2, 3] += 0.1
        clear_q, ik_success = self.get_ik_sol(clear_stacks_xyz)
        arm.safe_move_to_position(clear_q)

    ###############################
    #  Dynamic Trajectory Helpers #
    ###############################

    def get_block_trajectory(self):
        """
        Returns the block id, robot configuration s.t. end effect is at center of block, and time to grasp the block.

        INPUTS:
        None

        OUTPUTS:
        block_id - string, block id of the block to grasp
        target_block_xyz - 4x4 numpy array, robot configuration s.t. end effect is at center of block
        tf - scalar, time at grasping
        """
        # Detect block initial positions and orientations
        blocks_detected = False
        while not blocks_detected:
            blocks_i = detector.get_detections()
            ti = time_in_seconds()
            print("Got detection...", blocks_i)

            if blocks_i:
                print("Dynamic blocks detected")
                print(f"block pose list: {blocks_i}")

                block_dist_list = [
                    np.linalg.norm(H_o_cam[:3, 3]) for block_id, H_o_cam in blocks_i
                ]
                block_id_index = block_dist_list.index(min(block_dist_list))
                block_id, H_o_cam_i = blocks_i[block_id_index]
                if (
                    np.linalg.norm(H_o_cam_i[:3, 3]) < self.max_dist
                    and np.linalg.norm(H_o_cam_i[:3, 3]) > self.min_dist
                ):
                    print("Block within max distance")
                    blocks_detected = True
                else:
                    print("Block outside max distance")

        H_cam_ee = detector.get_H_ee_camera()
        q = arm.get_positions()
        H_ee_b = fk.forward(q)[1]  # Can hard code

        H_b_0 = np.identity(4)
        H_b_0[0:3, 3] = self.robot_center

        H_o_0_i = H_b_0 @ H_ee_b @ H_cam_ee @ H_o_cam_i
        H_o_0_f = self.z_rotation_matrix(self.d_theta) @ H_o_0_i

        H_0_b = H_b_0
        H_0_b[1, 3] = -H_0_b[1, 3]

        # Object to robot base at grasping
        H_o_b_f = H_0_b @ H_o_0_f

        # Correct the transformation matrix
        block_ori = correct_R(copy.deepcopy(H_o_b_f[0:3, 0:3]))
        block_pos = copy.deepcopy(H_o_b_f[0:3, 3]) + np.array(
            [self.grab_offset_x, self.grab_offset_y, 0]
        )
        updated_H_o_b_f = np.hstack(
            [block_ori.reshape((3, 3)), block_pos.reshape((3, 1))]
        )
        updated_H_o_b_f = np.vstack([updated_H_o_b_f, np.array([0, 0, 0, 1])])

        # Return grasping time
        tf = ti + self.d_theta / self.omega

        return block_id, updated_H_o_b_f, tf

    def z_rotation_matrix(self, angle):
        """
        Returns the rotation matrix for rotating an angle about z axis.

        INPUTS:
        angle - scalar, angle of rotation in rad

        OUTPUTS:
        T_rot - 4x4 numpy array, rotation matrix about z axis
        """

        T_rot = np.array(
            [
                [cos(angle), -sin(angle), 0, 0],
                [sin(angle), cos(angle), 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        )
        return T_rot


###########################
#  Transformation Helpers #
###########################


def correct_R(R):
    """
    Resolves the orientation of block so that z-axis is negative world-z
    """
    # print("Input R: ", R)
    # Find the Z-axis and correct small values to 0
    z_index = np.where(abs(R) == abs(R).max())
    z_index = (z_index[0][0], z_index[1][0])
    R[z_index[0], :] = 0
    R[:, z_index[1]] = 0

    # Get new X-col (Extract out remaining X and Y axis values)
    xy = np.delete(R, z_index[0], 0)
    xy = np.delete(xy, z_index[1], 1)
    x1 = xy[0, 0]
    x2 = xy[1, 0]
    x_col = np.array([x1, x2, 0])

    if x1 < 0:
        x_col = x_col * -1

    # Get new Y-col by crossing Z-col with X-col
    z_col = np.array([0, 0, -1])
    y_col = np.cross(z_col, x_col)
    R_temp = np.hstack(
        [x_col.reshape((3, -1)), y_col.reshape((3, -1)), z_col.reshape((3, -1))]
    )

    # Normalize matrix before output
    R_new = Rot.from_matrix(R_temp)
    R_new = R_new.as_matrix()
    # print("New R", R_new)
    return R_new


def trans(d):
    """
    Compute pure translation homogenous transformation
    """
    return np.array(
        [
            [1, 0, 0, d[0]],
            [0, 1, 0, d[1]],
            [0, 0, 1, d[2]],
            [0, 0, 0, 1],
        ]
    )


def roll(a):
    """
    Compute homogenous transformation for rotation around x axis by angle a
    """
    return np.array(
        [
            [1, 0, 0, 0],
            [0, cos(a), -sin(a), 0],
            [0, sin(a), cos(a), 0],
            [0, 0, 0, 1],
        ]
    )


def pitch(a):
    """
    Compute homogenous transformation for rotation around y axis by angle a
    """
    return np.array(
        [
            [cos(a), 0, -sin(a), 0],
            [0, 1, 0, 0],
            [sin(a), 0, cos(a), 0],
            [0, 0, 0, 1],
        ]
    )


def yaw(a):
    """
    Compute homogenous transformation for rotation around z axis by angle a
    """
    return np.array(
        [
            [cos(a), -sin(a), 0, 0],
            [sin(a), cos(a), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )


def transform(d, rpy):
    """
    Helper function to compute a homogenous transform of a translation by d and
    rotation corresponding to roll-pitch-yaw euler angles
    """
    return trans(d) @ roll(rpy[0]) @ pitch(rpy[1]) @ yaw(rpy[2])


#############################
##          MAIN           ##
#############################


if __name__ == "__main__":
    try:
        team = rospy.get_param("team")  # 'red' or 'blue'
    except KeyError:
        print("Team must be red or blue - make sure you are running final.launch!")
        exit()

    print("team", team)

    rospy.init_node("team_script")
    arm = ArmController()
    detector = ObjectDetector()
    ik = IK()
    fk = FK()

    start_position = np.array(
        [
            -0.01779206,
            -0.76012354,
            0.01978261,
            -2.34205014,
            0.02984053,
            1.54119353 + pi / 2,
            0.75344866,
        ]
    )
    arm.safe_move_to_position(start_position)  # on your mark!

    print("\n****************")
    if team == "blue":
        print("** BLUE TEAM  **")
        # Params/dimensions/settings for BLUE TEAM:
        # Fixed/given params:
        # Robot center in WORLD frame:
        robot_center = np.array([0, 0.990, 0])
        # Center of dynamic table (BLUE frame):
        dynamic_table_center = np.array([0, 0, 0.2]) - robot_center

        # Joint configts for known positions:
        goal_config = np.array(
            [-0.1681, 0.24085, -0.16154, -1.04058, 0.04007, 1.27863, 0.47189]
        )
        start_static_config = np.array(
            [
                0.13359178,
                -0.04639921,
                0.17513385,
                -1.80883275,
                0.00823362,
                1.76313594,
                1.09236519,
            ]
        )
        start_dynamic_config = np.array(
            [0.18216, -0.77313, -1.74074, -2.03142, -0.11266, 2.81834, -0.43331]
        )  # BLUE DYNAMIC CONFIG

        # Gripper offsets #TODO: get these values in the lab
        grab_offset_x = 0.0
        grab_offset_y = 0.0
        pre_stack_config = np.array(
            [
                [
                    -0.12017144,
                    0.20148316,
                    -0.1791519,
                    -2.02175864,
                    0.04477032,
                    2.21961892,
                    0.46256024,
                ],
                [
                    -0.12870565,
                    0.14354317,
                    -0.17029047,
                    -1.9625534,
                    0.02815199,
                    2.10387463,
                    0.47381099,
                ],
                [
                    -0.1348747,
                    0.10292604,
                    -0.16345086,
                    -1.88331617,
                    0.01826295,
                    1.98481842,
                    0.48057464,
                ],
                [
                    -0.13865305,
                    0.08056227,
                    -0.15942016,
                    -1.78296791,
                    0.01333892,
                    1.86248844,
                    0.48399722,
                ],
                [
                    -0.14068104,
                    0.07780266,
                    -0.15842672,
                    -1.65900843,
                    0.01243136,
                    1.73582808,
                    0.48471935,
                ],
                [
                    -0.14255396,
                    0.09731537,
                    -0.16018475,
                    -1.50608521,
                    0.01550562,
                    1.60215866,
                    0.48291829,
                ],
                [
                    -0.14786741,
                    0.14533554,
                    -0.16350365,
                    -1.3120031,
                    0.02373386,
                    1.45545915,
                    0.47845262,
                ],
                [
                    -0.1680972,
                    0.24084678,
                    -0.16154062,
                    -1.04057937,
                    0.04007258,
                    1.27862971,
                    0.47189338,
                ],
            ]
        )

        stack_config = np.array(
            [
                [
                    -0.11617387,
                    0.22929824,
                    -0.18300623,
                    -2.03991859,
                    0.05383773,
                    2.26480443,
                    0.4564503,
                ],
                [
                    -0.12553592,
                    0.16469134,
                    -0.17367675,
                    -1.98860928,
                    0.03387221,
                    2.15060359,
                    0.46992717,
                ],
                [
                    -0.13271013,
                    0.11703761,
                    -0.1658759,
                    -1.91746121,
                    0.02154065,
                    2.03281412,
                    0.47832336,
                ],
                [
                    -0.13740244,
                    0.08724789,
                    -0.16067135,
                    -1.8257576,
                    0.01479285,
                    1.91185284,
                    0.48297683,
                ],
                [
                    -0.14000966,
                    0.07642833,
                    -0.15846117,
                    -1.71167773,
                    0.01233644,
                    1.78713622,
                    0.48473403,
                ],
                [
                    -0.1416387,
                    0.08656356,
                    -0.15921931,
                    -1.57127907,
                    0.01375851,
                    1.65674498,
                    0.4839452,
                ],
                [
                    -0.14489146,
                    0.12200782,
                    -0.16219716,
                    -1.39591982,
                    0.01968426,
                    1.51635234,
                    0.48056577,
                ],
                [
                    -0.15664908,
                    0.19429626,
                    -0.16410936,
                    -1.16344329,
                    0.03229602,
                    1.35530195,
                    0.47458217,
                ],
            ]
        )
        move_clear_config = np.array(
            [
                [
                    -0.12877964,
                    0.14354134,
                    -0.17021329,
                    -1.96255347,
                    0.028139,
                    2.1038749,
                    0.47381997,
                ],
                [
                    -0.13487503,
                    0.10292603,
                    -0.16345052,
                    -1.88331617,
                    0.01826291,
                    1.98481842,
                    0.48057466,
                ],
                [
                    -0.13865305,
                    0.08056227,
                    -0.15942015,
                    -1.78296791,
                    0.01333892,
                    1.86248844,
                    0.48399722,
                ],
                [
                    -0.14068104,
                    0.07780266,
                    -0.15842672,
                    -1.65900843,
                    0.01243136,
                    1.73582808,
                    0.48471935,
                ],
                [
                    -0.14255396,
                    0.09731537,
                    -0.16018475,
                    -1.50608521,
                    0.01550562,
                    1.60215866,
                    0.48291829,
                ],
                [
                    -0.14786741,
                    0.14533554,
                    -0.16350365,
                    -1.3120031,
                    0.02373386,
                    1.45545915,
                    0.47845262,
                ],
                [
                    -0.1680972,
                    0.24084678,
                    -0.16154062,
                    -1.04057937,
                    0.04007258,
                    1.27862971,
                    0.47189338,
                ],
                [
                    -0.21833717,
                    0.38335346,
                    -0.13106607,
                    -0.68788684,
                    0.05578857,
                    1.07117015,
                    0.4722719,
                ],
            ]
        )

        dy_pre_grab = np.array(
            [
                -2.14715840916761,
                0.8513986422029582,
                -0.07226399844978096,
                -1.1442969291071703,
                1.0700673392869784,
                1.4529516432063128,
                -1.391274705531255,
            ]
        )
        dy_pre_grab_2 = np.array(
            [
                -2.1051486308850613,
                0.9694577902539792,
                0.025312944109398532,
                -1.0447171674933353,
                1.0482268134522645,
                1.5715325655677292,
                -1.345128525688214,
            ]
        )
        dy_grab = np.array(
            [
                -1.8030987659425644,
                0.8708581711216665,
                -0.011277971710271506,
                -1.1814284693932218,
                1.0093227948797712,
                1.6168474571809974,
                -1.31901600375775,
            ]
        )
        dy_post_grab = np.array(
            [
                -0.2905346797182791,
                -0.6118008448241433,
                -0.7635707116473287,
                -1.765392217425988,
                0.48839098802196934,
                1.8554765748400242,
                -0.29809889337237405,
            ]
        )
        dy_post_stack = np.array(
            [
                -0.6089358785729624,
                -1.3504754586001617,
                -0.309160982929769,
                -2.4862705522318267,
                0.40703252206579627,
                1.9297159233586592,
                -0.007820929548253032,
            ]
        )

        dy_pre_stack = [
            np.array(
                [
                    0.022546184296116146,
                    0.09108242582540114,
                    -0.3071773015379963,
                    -2.299412722307401,
                    2.378026683482703,
                    2.7625940613803412,
                    -1.749544764503615,
                ]
            ),
            np.array(
                [
                    -0.13301719667641088,
                    -0.002536559257853882,
                    -0.12393187657160448,
                    -2.2870679458618186,
                    2.2298304613471767,
                    2.8596911035286676,
                    -1.594928428100942,
                ]
            ),
            np.array(
                [
                    -0.23093515251968003,
                    -0.08472672984743908,
                    0.0008593032813257182,
                    -2.284681661371483,
                    2.006433867329168,
                    2.9219430151198535,
                    -1.3469308593026996,
                ]
            ),
            np.array(
                [
                    -0.205749787272092,
                    -0.1607888559502724,
                    -0.004983917067845191,
                    -2.2915478006085683,
                    1.7140561264611105,
                    2.957993117226204,
                    -1.0382877938555741,
                ]
            ),
            np.array(
                [
                    -0.08341798806273644,
                    -0.2324401058156819,
                    -0.11797704832335253,
                    -2.3022736461673237,
                    1.3511513324722861,
                    2.980079686556515,
                    -0.686718372822428,
                ]
            ),
            np.array(
                [
                    -0.016756060198561313,
                    -0.29562552654550767,
                    -0.19927466124140714,
                    -2.297595397223644,
                    0.9660468963317002,
                    2.9590143654069068,
                    -0.32797261445646864,
                ]
            ),
            np.array(
                [
                    -0.07056698930380925,
                    -0.3316125027108219,
                    -0.17620386583301656,
                    -2.2418991788114835,
                    0.7263046630935128,
                    2.8753047117838575,
                    -0.09335787168120904,
                ]
            ),
            np.array(
                [
                    -0.11863834773009037,
                    -0.34666995918002963,
                    -0.15289371489010095,
                    -2.150000240535441,
                    0.5832584685527085,
                    2.7726601415880334,
                    0.05478772470142591,
                ]
            ),
        ]
        dy_stack = [
            np.array(
                [
                    0.08563609611285851,
                    0.1345966500202734,
                    -0.3796589295172232,
                    -2.305152709775177,
                    2.420460148667199,
                    2.715390397647481,
                    -1.7854955344833257,
                ]
            ),
            np.array(
                [
                    -0.07329887523840385,
                    0.03287287298550025,
                    -0.19514566879262812,
                    -2.2912137691264496,
                    2.2975129163729746,
                    2.8248339570639196,
                    -1.6685270689293015,
                ]
            ),
            np.array(
                [
                    -0.20409907449758657,
                    -0.052628437518235587,
                    -0.03604269326482698,
                    -2.2842014717123296,
                    2.104866775134194,
                    2.9011321576851663,
                    -1.4558147861793116,
                ]
            ),
            np.array(
                [
                    -0.2313201491174345,
                    -0.13110601111515763,
                    0.013867088549175538,
                    -2.288104271569686,
                    1.8386928863846719,
                    2.945544926682348,
                    -1.1667732339182428,
                ]
            ),
            np.array(
                [
                    -0.1380210320046049,
                    -0.20399626249996655,
                    -0.06543479863671418,
                    -2.29783814794239,
                    1.505718466437234,
                    2.973245408446295,
                    -0.8330590384305492,
                ]
            ),
            np.array(
                [
                    -0.021840559818762453,
                    -0.27316638656410863,
                    -0.1844404210948603,
                    -2.3047541341357465,
                    1.1085493244684916,
                    2.9768874815444764,
                    -0.4615987403793962,
                ]
            ),
            np.array(
                [
                    -0.045961887459023555,
                    -0.3201281355815583,
                    -0.18874298400017225,
                    -2.269652386047179,
                    0.8057912187337852,
                    2.912769646786047,
                    -0.17282084782264323,
                ]
            ),
            np.array(
                [
                    -0.10233413814062106,
                    -0.3428776342211688,
                    -0.16027134794494985,
                    -2.1901256100081294,
                    0.633153173258424,
                    2.8147062825048708,
                    0.0022448492603602314,
                ]
            ),
        ]
        dy_clear = [
            np.array(
                [
                    -0.13301554760056308,
                    -0.002536559553695275,
                    -0.12393352501537779,
                    -2.2870679448458526,
                    2.229830463249834,
                    2.8596911065613044,
                    -1.5949284329415765,
                ]
            ),
            np.array(
                [
                    -0.23093515206943424,
                    -0.0847267298456056,
                    0.0008593028410677011,
                    -2.2846816613632956,
                    2.006433867339038,
                    2.9219430151509056,
                    -1.3469308593367035,
                ]
            ),
            np.array(
                [
                    -0.2057497872706635,
                    -0.16078885595026987,
                    -0.004983917069190823,
                    -2.2915478006085497,
                    1.7140561264610552,
                    2.9579931172264042,
                    -1.038287793855649,
                ]
            ),
            np.array(
                [
                    -0.08341798806273568,
                    -0.23244010581568192,
                    -0.11797704832335326,
                    -2.3022736461673228,
                    1.351151332472287,
                    2.9800796865565156,
                    -0.6867183728224291,
                ]
            ),
            np.array(
                [
                    -0.01675606019856284,
                    -0.29562552654550833,
                    -0.1992746612414065,
                    -2.297595397223643,
                    0.9660468963316973,
                    2.9590143654069068,
                    -0.3279726144564658,
                ]
            ),
            np.array(
                [
                    -0.0705669893038091,
                    -0.3316125027108223,
                    -0.1762038658330172,
                    -2.2418991788114844,
                    0.7263046630935138,
                    2.875304711783858,
                    -0.09335787168120999,
                ]
            ),
            np.array(
                [
                    -0.11863834773009041,
                    -0.34666995918002924,
                    -0.15289371489010115,
                    -2.150000240535441,
                    0.5832584685527087,
                    2.7726601415880334,
                    0.054787724701425726,
                ]
            ),
            np.array(
                [
                    -0.14391952419063828,
                    -0.3434864264864442,
                    -0.14490938729411665,
                    -2.0327784925874077,
                    0.48710016980436,
                    2.664042656562759,
                    0.15995346037992522,
                ]
            ),
        ]

        dy_grab_HeeW = np.array(
            [
                [5.00000000e-01, 0, 8.66025404e-01, 0],
                [0, -1.0, -5.37967580e-12, -7.20000000e-01],
                [8.66025404e-01, 0, -5.00000000e-01, 2.25000000e-01],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        start_static_config_alt = np.array(
            [
                0.040589678124845166,
                -1.0125697600844437,
                0.4353578531533338,
                -2.67959559914851,
                0.22586343850626978,
                2.209569254346148,
                1.072480639851587,
            ]
        )

    else:
        print("**  RED TEAM  **")
        # Params/dimensions/settings for RED TEAM:
        # Fixed/given params:
        # Robot center in WORLD frame:
        robot_center = np.array([0, -0.990, 0])
        # Center of dynamic table (RED frame):
        dynamic_table_center = np.array([0, 0, 0.2]) - robot_center

        # Joint configts for known positions:
        goal_config = np.array(
            [0.25669, 0.23861, 0.04606, -1.04074, -0.01136, 1.27912, 1.08358]
        )
        start_static_config = np.array(
            [
                -0.16161129,
                -0.04618842,
                -0.14780904,
                -1.80883488,
                -0.00692763,
                1.76314489,
                0.47745752,
            ]
        )
        start_dynamic_config = np.array(
            [0.63928, 0.29078, 0.84717, -1.95899, 0.95819, 2.60663, -1.13571]
        )
        # RED DYNAMIC CONFIG

        # Gripper offsets #TODO: get these values in the lab
        grab_offset_x = 0.0
        grab_offset_y = 0.0

        pre_stack_config = np.array(
            [
                [
                    0.24643033,
                    0.19858653,
                    0.04757504,
                    -2.02193556,
                    -0.01178106,
                    2.22026982,
                    1.08559489,
                ],
                [
                    0.21587639,
                    0.14197329,
                    0.0794212,
                    -1.96261356,
                    -0.0130368,
                    2.10410807,
                    1.08652801,
                ],
                [
                    0.19526785,
                    0.10210072,
                    0.100649,
                    -1.88333674,
                    -0.01118695,
                    1.98490125,
                    1.08529581,
                ],
                [
                    0.18428433,
                    0.08006044,
                    0.11199765,
                    -1.78297708,
                    -0.00933275,
                    1.86252606,
                    1.08400864,
                ],
                [
                    0.18303628,
                    0.07735094,
                    0.11409394,
                    -1.65901615,
                    -0.0089187,
                    1.73585972,
                    1.08365568,
                ],
                [
                    0.19257984,
                    0.09666226,
                    0.10657036,
                    -1.5060993,
                    -0.01027109,
                    1.60221489,
                    1.08437727,
                ],
                [
                    0.21571894,
                    0.14409567,
                    0.08652857,
                    -1.31204592,
                    -0.01249289,
                    1.45561879,
                    1.08531751,
                ],
                [
                    0.25668889,
                    0.23861267,
                    0.04606008,
                    -1.04073549,
                    -0.01136283,
                    1.27912236,
                    1.0835763,
                ],
            ]
        )

        stack_config = np.array(
            [
                [
                    0.26157402,
                    0.22570225,
                    0.03174743,
                    -2.04018367,
                    -0.00924863,
                    2.26575498,
                    1.08383743,
                ],
                [
                    0.22689694,
                    0.16266508,
                    0.0679689,
                    -1.98870269,
                    -0.0131512,
                    2.15095909,
                    1.08657873,
                ],
                [
                    0.20233909,
                    0.11597564,
                    0.09339491,
                    -1.91749239,
                    -0.01205656,
                    2.03293808,
                    1.0858842,
                ],
                [
                    0.1875308,
                    0.08665567,
                    0.10859166,
                    -1.82576958,
                    -0.00995347,
                    1.91190174,
                    1.08444617,
                ],
                [
                    0.18232317,
                    0.07598455,
                    0.11435969,
                    -1.71168524,
                    -0.00886922,
                    1.78716711,
                    1.08365806,
                ],
                [
                    0.18730113,
                    0.0860276,
                    0.110876,
                    -1.57128406,
                    -0.00954253,
                    1.6567821,
                    1.08398814,
                ],
                [
                    0.20459324,
                    0.12106073,
                    0.09637181,
                    -1.39594628,
                    -0.0116378,
                    1.51645438,
                    1.08503006,
                ],
                [
                    0.23786941,
                    0.19247175,
                    0.06562995,
                    -1.16353572,
                    -0.01284154,
                    1.35562044,
                    1.08494695,
                ],
            ]
        )

        move_clear_config = np.array(
            [
                [
                    0.21580192,
                    0.14197413,
                    0.0794988,
                    -1.96261353,
                    -0.01304959,
                    2.10410795,
                    1.08653686,
                ],
                [
                    0.19526772,
                    0.10210072,
                    0.10064912,
                    -1.88333674,
                    -0.01118696,
                    1.98490125,
                    1.08529582,
                ],
                [
                    0.18428433,
                    0.08006044,
                    0.11199765,
                    -1.78297708,
                    -0.00933275,
                    1.86252606,
                    1.08400864,
                ],
                [
                    0.18303628,
                    0.07735094,
                    0.11409394,
                    -1.65901615,
                    -0.0089187,
                    1.73585972,
                    1.08365568,
                ],
                [
                    0.19257984,
                    0.09666226,
                    0.10657036,
                    -1.5060993,
                    -0.01027109,
                    1.60221489,
                    1.08437727,
                ],
                [
                    0.21571894,
                    0.14409567,
                    0.08652857,
                    -1.31204592,
                    -0.01249289,
                    1.45561879,
                    1.08531751,
                ],
                [
                    0.25668889,
                    0.23861267,
                    0.04606008,
                    -1.04073549,
                    -0.01136283,
                    1.27912236,
                    1.0835763,
                ],
                [
                    0.29475255,
                    0.38256049,
                    -0.00469743,
                    -0.68618803,
                    0.00200042,
                    1.07097963,
                    1.07675552,
                ],
            ]
        )

        dy_pre_grab = np.array(
            [
                0.620713314323936,
                0.894738617575305,
                0.7397801191585932,
                -0.7783644122098183,
                0.5688844762939824,
                1.3948335469174074,
                -0.8889964041662688,
            ]
        )
        dy_pre_grab_2 = np.array(
            [
                0.673192569671536,
                1.0777817054757857,
                0.6956041674938864,
                -1.086398735367818,
                0.5097821263929878,
                1.7715138875932415,
                -1.186385720538466,
            ]
        )
        dy_grab = np.array(
            [
                0.8862704347273429,
                1.0018698991759956,
                0.7357053433940908,
                -1.226634998173651,
                0.4423113036884306,
                1.8394318387860524,
                -1.119083825548751,
            ]
        )
        dy_post_grab = np.array(
            [
                0.6082024991055888,
                -0.6466068531433944,
                0.2162936402031406,
                -2.021240114654672,
                0.9124041994617234,
                2.1214530976262256,
                -0.1400549474831236,
            ]
        )
        dy_post_stack = np.array(
            [
                -0.22225376986598464,
                0.2079386995800755,
                0.220816132815708,
                -1.0221473172360114,
                1.0308038871762426,
                1.3822725747315425,
                -0.4935882989068451,
            ]
        )

        dy_pre_stack = [
            np.array(
                [
                    -0.12027511266751909,
                    0.27307663952710615,
                    0.16293683984745622,
                    -2.0834325102632074,
                    1.1683258678241464,
                    1.983788148658302,
                    -1.473161938796602,
                ]
            ),
            np.array(
                [
                    -0.08971789248492266,
                    0.19024983715754143,
                    0.12188770541662695,
                    -2.0688576264986835,
                    1.146818168714557,
                    1.9280495037898566,
                    -1.3869372424444646,
                ]
            ),
            np.array(
                [
                    -0.06101171853867615,
                    0.12163536719549473,
                    0.08290662895779127,
                    -2.0373919568763896,
                    1.1233630464053406,
                    1.8727874162645528,
                    -1.2984013857097927,
                ]
            ),
            np.array(
                [
                    -0.03637649899760683,
                    0.06785975700871041,
                    0.048687051915317886,
                    -1.9889333110510672,
                    1.1001744183339495,
                    1.8174745898270528,
                    -1.2086387955216835,
                ]
            ),
            np.array(
                [
                    -0.017777269446539665,
                    0.029394327911407885,
                    0.02127733145637896,
                    -1.92324135776752,
                    1.0793500092958033,
                    1.7612639818499713,
                    -1.118393128929258,
                ]
            ),
            np.array(
                [
                    -0.007188989387413035,
                    0.006725929330887612,
                    0.002374736400965154,
                    -1.8397405081150175,
                    1.0626797254754448,
                    1.703271251107172,
                    -1.027911546271316,
                ]
            ),
            np.array(
                [
                    -0.007020604638979065,
                    0.0005580639692901799,
                    -0.006098741308835431,
                    -1.7371995424850244,
                    1.0515611492169923,
                    1.6427478536722613,
                    -0.9367559544417976,
                ]
            ),
            np.array(
                [
                    -0.020767985503666056,
                    0.012156317712665968,
                    -0.000984165822287851,
                    -1.613163540075079,
                    1.0468615353404243,
                    1.579235744098171,
                    -0.8434876072264379,
                ]
            ),
        ]

        dy_stack = [
            np.array(
                [
                    -0.13264956137504286,
                    0.3099941599160781,
                    0.17941773633614652,
                    -2.0845548293888987,
                    1.1758727344319808,
                    2.0062583613707172,
                    -1.506703577722431,
                ]
            ),
            np.array(
                [
                    -0.1018661529983567,
                    0.22171998152872338,
                    0.13824574389655325,
                    -2.076710510804881,
                    1.155773831730826,
                    1.9502796167584175,
                    -1.4217724327521355,
                ]
            ),
            np.array(
                [
                    -0.07212128232289634,
                    0.14733014732264763,
                    0.09806141078650014,
                    -2.0520101315491064,
                    1.1328376284490105,
                    1.8948576683652842,
                    -1.3340177048067072,
                ]
            ),
            np.array(
                [
                    -0.04561213124267729,
                    0.08755772239330974,
                    0.061659257828330365,
                    -2.0103663448165854,
                    1.109278161505271,
                    1.839656712256697,
                    -1.2446340736977326,
                ]
            ),
            np.array(
                [
                    -0.024370477839989493,
                    0.042914464764732575,
                    0.03131269831331334,
                    -1.9516124759136253,
                    1.0872733772502357,
                    1.7839140992141733,
                    -1.1545205094762783,
                ]
            ),
            np.array(
                [
                    -0.010325380788897795,
                    0.0138604436407955,
                    0.008806575126576073,
                    -1.8753370593232286,
                    1.0687503506526908,
                    1.7267347215568356,
                    -1.064142459150471,
                ]
            ),
            np.array(
                [
                    -0.005644464350118053,
                    0.000982596667336327,
                    -0.0041223624193832235,
                    -1.7806210027411877,
                    1.05527168357024,
                    1.6672990491064668,
                    -0.9733633749171597,
                ]
            ),
            np.array(
                [
                    -0.01325393664449503,
                    0.005263197854574342,
                    -0.005011353721836761,
                    -1.6655932358645016,
                    1.047949540471944,
                    1.6050091475967194,
                    -0.8811983646155379,
                ]
            ),
        ]

        dy_clear = [
            np.array(
                [
                    -0.08964219551352211,
                    0.1902483179669467,
                    0.12180730286529412,
                    -2.0688565032170496,
                    1.1468294843686577,
                    1.9280380084677424,
                    -1.3869417366509083,
                ]
            ),
            np.array(
                [
                    -0.06101164224639892,
                    0.12163536657955747,
                    0.08290654912514198,
                    -2.0373919561323635,
                    1.1233630537341166,
                    1.8727874089421,
                    -1.2984013887034314,
                ]
            ),
            np.array(
                [
                    -0.03637649890801279,
                    0.06785975700851674,
                    0.04868705182303815,
                    -1.988933311050566,
                    1.1001744183388231,
                    1.817474589822317,
                    -1.2086387955237066,
                ]
            ),
            np.array(
                [
                    -0.017777269446415105,
                    0.029394327911407753,
                    0.02127733145625244,
                    -1.9232413577675198,
                    1.0793500092958062,
                    1.7612639818499682,
                    -1.118393128929259,
                ]
            ),
            np.array(
                [
                    -0.007188989387412714,
                    0.006725929330887722,
                    0.0023747364009648543,
                    -1.8397405081150173,
                    1.0626797254754448,
                    1.703271251107172,
                    -1.027911546271316,
                ]
            ),
            np.array(
                [
                    -0.007020604638978996,
                    0.0005580639692902331,
                    -0.006098741308835464,
                    -1.7371995424850246,
                    1.051561149216992,
                    1.6427478536722615,
                    -0.9367559544417978,
                ]
            ),
            np.array(
                [
                    -0.020767985503666073,
                    0.012156317712666146,
                    -0.0009841658222876975,
                    -1.6131635400750788,
                    1.046861535340424,
                    1.5792357440981708,
                    -0.8434876072264377,
                ]
            ),
            np.array(
                [
                    -0.0543203580696233,
                    0.044119951442890847,
                    0.024414637788424266,
                    -1.4627589384707822,
                    1.0482491314734137,
                    1.5128980739775795,
                    -0.7449579520967025,
                ]
            ),
        ]

        dy_grab_HeeW = np.array(
            [
                [-5.00000000e-01, 0, -8.66025404e-01, 0],
                [0, 1.0, 0, 7.20000000e-01],
                [8.66025404e-01, 0, -5.00000000e-01, 2.25000000e-01],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        start_static_config_alt = np.array(
            [
                -0.4157854388731539,
                -0.920091194192555,
                -0.14293951983243683,
                -2.6815223789511,
                0.17782519313141035,
                2.2171777742159153,
                0.18143271637236488,
            ]
        )

    # TUPLE of robot params to pass into code
    # (ALL IN RESPECTIVE ROBOT FRAMES, except robot_center):
    robot_params = (
        robot_center,
        dynamic_table_center,
        goal_config,
        start_static_config,
        start_dynamic_config,
        grab_offset_x,
        grab_offset_y,
        pre_stack_config,
        stack_config,
        move_clear_config,
        dy_pre_grab,
        dy_pre_grab_2,
        dy_grab,
        dy_post_grab,
        dy_pre_stack,
        dy_stack,
        dy_clear,
        dy_grab_HeeW,
        start_static_config_alt,
        dy_post_stack)

    print("****************")
    input("\nWaiting for start... Press ENTER to begin!\n")  # get set!
    print("Go!\n")  # go!

    routine = MAIN_ROUTINE(robot_params)
    routine.go_static()
    # routine.go_dynamic_traj()
    routine.go_dynamic_wait_grab()
