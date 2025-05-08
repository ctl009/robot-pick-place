import numpy as np
from numpy import pi

# from lib.calculateFK import FK


class FK:
    def __init__(self):
        # TODO: you may want to define geometric parameters here that will be
        # useful in computing the forward kinematics. The data you will need
        # is provided in the lab handout

        # Joint Angles
        self._q = [0, 0, 0, pi, 0, pi, -pi / 4]

        # Link Lengths
        self._d = [0.333, 0, 0.316, 0, 0.384, 0, 0.210]

        # Link Twists
        self._alpha = [-pi / 2, pi / 2, pi / 2, pi / 2, -pi / 2, pi / 2, 0]

        # Link Offsets
        self._a = [0, 0, 0.0825, 0.0825, 0, 0.088, 0]

    def calculate_cons_trans_mat(self, q):
        q = list(map(sum, zip(self._q, q)))

        transform_list = {
            "T01": [],
            "T12": [],
            "T23": [],
            "T34": [],
            "T45": [],
            "T56": [],
            "T6e": [],
        }

        i = 0
        for name in transform_list.keys():
            a = self._a[i]
            alpha = self._alpha[i]
            d = self._d[i]
            theta = q[i]

            matrix = np.array(
                [
                    [
                        np.cos(theta),
                        -np.sin(theta) * np.cos(alpha),
                        np.sin(theta) * np.sin(alpha),
                        a * np.cos(theta),
                    ],
                    [
                        np.sin(theta),
                        np.cos(theta) * np.cos(alpha),
                        -np.cos(theta) * np.sin(alpha),
                        a * np.sin(theta),
                    ],
                    [0, np.sin(alpha), np.cos(alpha), d],
                    [0, 0, 0, 1],
                ]
            )
            transform_list[name] = matrix
            i += 1

        cons_trans_list = {
            "T01": [],
            "T02": [],
            "T03": [],
            "T04": [],
            "T05": [],
            "T06": [],
            "T0e": [],
        }
        ref_con = {
            "T01": 1,
            "T02": 2,
            "T03": 3,
            "T04": 4,
            "T05": 5,
            "T06": 6,
            "T0e": 7,
        }
        for name in cons_trans_list.keys():
            matrix = np.eye(4)
            for i in range(ref_con[name]):
                matrix = np.matmul(matrix, list(transform_list.values())[i])
            cons_trans_list[name] = matrix

        # # Verifying Transformation Matrices
        # for name in transform_list.keys():
        #     print(name)
        #     print(transform_list[name])

        # # Verifying Consecutive Transformation Matrices
        # for name in cons_trans_list.keys():
        #     print(name)
        #     print(cons_trans_list[name])

        return cons_trans_list

    def forward(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        jointPositions -8 x 3 matrix, where each row corresponds to a rotational joint of the robot or end effector
                  Each row contains the [x,y,z] coordinates in the world frame of the respective joint's center in meters.
                  The base of the robot is located at [0,0,0].
        T0e       - a 4 x 4 homogeneous transformation matrix,
                  representing the end effector frame expressed in the
                  world frame
        """

        # Your Lab 1 code starts here

        jointPositions = np.zeros((8, 3))
        T0e = np.identity(4)

        jointPositions = [
            # [0, 0, 0.141],
            [0, 0, 0],
            [0, 0, 0.195],
            [0, 0, 0],
            [0, 0, 0.125],
            [0, 0, -0.015],
            [0, 0, 0.051],
            [0, 0, 0],
        ]

        ref_con = {
            "T01": 1,
            "T02": 2,
            "T03": 3,
            "T04": 4,
            "T05": 5,
            "T06": 6,
            "T0e": 7,
        }

        cons_trans_list = self.calculate_cons_trans_mat(q)

        def get_key_by_val(val):
            for key, value in ref_con.items():
                if value == val:
                    return key
            return None

        for i, elements in enumerate(jointPositions):
            elements = np.append(elements, 1)
            key = get_key_by_val(i + 1)
            jointPositions[i] = np.matmul(cons_trans_list[key], elements)[:-1]

        jointPositions.insert(0, np.array([0, 0, 0.141]))
        jointPositions = np.array(jointPositions)
        T0e = cons_trans_list["T0e"]

        # Your code ends here

        return jointPositions, T0e

    # feel free to define additional helper methods to modularize your solution for lab 1

    # This code is for Lab 2, you can ignore it ofr Lab 1
    def get_axis_of_rotation(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        axis_of_rotation_list: - 3x7 np array of unit vectors describing the axis of rotation for each joint in the
                                 world frame

        """
        # STUDENT CODE HERE: This is a function needed by lab 2
        cons_trans_list = self.calculate_cons_trans_mat(q)

        axis_of_rotation_list = [[0, 0, 1]]

        for matrix in cons_trans_list.values():
            matrix = np.array(matrix)
            axis_of_rotation_list.append(matrix[:3, 2])

        return np.transpose(np.array(axis_of_rotation_list)[:-1])

    def compute_Ai(self, q):
        """
        INPUT:
        q - 1x7 vector of joint angles [q0, q1, q2, q3, q4, q5, q6]

        OUTPUTS:
        Ai: - 4x4 list of np array of homogenous transformations describing the FK of the robot. Transformations are not
              necessarily located at the joint locations
        """
        # STUDENT CODE HERE: This is a function needed by lab 2
        cons_trans_list = self.calculate_cons_trans_mat(q)

        Ai_list = []

        for matrix in cons_trans_list.values():
            Ai_list.append(np.array(matrix))

        # print(Ai_list[6])

        return Ai_list


def calcJacobian(q_in):
    """
    Calculate the full Jacobian of the end effector in a given configuration
    :param q_in: 1 x 7 configuration vector (of joint angles) [q1,q2,q3,q4,q5,q6,q7]
    :return: J - 6 x 7 matrix representing the Jacobian, where the first three
    rows correspond to the linear velocity and the last three rows correspond to
    the angular velocity, expressed in world frame coordinates
    """

    J = np.zeros((6, 7))

    ## STUDENT CODE GOES HERE

    calFK = FK()
    jointPos, T0e = calFK.forward(q_in)
    O_n = jointPos[-1]
    rot_axis = calFK.get_axis_of_rotation(q_in)
    # Ai_list = calFK.compute_Ai(q_in)

    for i in range(7):
        Jv = np.cross(rot_axis[:, i], (O_n - jointPos[i]))
        J[:3, i] = Jv
        J[3:6, i] = rot_axis[:, i]
    return J


if __name__ == "__main__":
    q = np.array([0, 0, 0, -np.pi / 2, 0, np.pi / 2, np.pi / 4])
    print(np.round(calcJacobian(q), 3))
