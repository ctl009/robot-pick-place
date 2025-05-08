import numpy as np

# import scipy
# from scipy.linalg import null_space
from lib.IK_velocity import IK_velocity
from lib.calcJacobian import calcJacobian

"""
Lab 3
"""


def IK_velocity_null(q_in, v_in, omega_in, b):
    """
    :param q_in: 1 x 7 vector corresponding to the robot's current configuration.
    :param v_in: The desired linear velocity in the world frame. If any element is
    Nan, then that velocity can be anything
    :param omega_in: The desired angular velocity in the world frame. If any
    element is Nan, then that velocity is unconstrained i.e. it can be anything
    :param b: 7 x 1 Secondary task joint velocity vector
    :return:
    dq + null - 1 x 7 vector corresponding to the joint velocities + secondary task null velocities
    """

    ## STUDENT CODE GOES HERE
    dq = np.zeros((1, 7))
    null = np.zeros((1, 7))
    b = b.reshape((7, 1))
    v_in = np.array(v_in)
    v_in = v_in.reshape((3, 1))
    omega_in = np.array(omega_in)
    omega_in = omega_in.reshape((3, 1))

    # ================================================================
    J = calcJacobian(q_in)  # 6x7
    v_mat = np.append(v_in, omega_in)  # 6x1
    to_remove = ~np.isnan(v_mat)  # 6x1

    # Filter NaN values
    v_mat_no_NaN = v_mat[to_remove]  # 6x1
    J_no_NaN = J[to_remove]  # (6)x7 for example

    pseu_J = np.matmul(
        J_no_NaN.T,
        np.linalg.inv(np.matmul(J_no_NaN, J_no_NaN.T)),
    )  # 7x(6)

    # =================================================================
    null = np.matmul(np.eye(7) - np.matmul(pseu_J, J_no_NaN), b)  # 7x1

    # =================================================================
    dq = np.matmul(pseu_J, v_mat_no_NaN)  # 7x1

    dq = dq.reshape((1, 7))
    null = null.reshape((1, 7))

    return dq + null
    # return dq


if __name__ == "__main__":
    q = np.array([0, 0, 0, -np.pi / 2, 0, np.pi / 2, np.pi / 4])
    v_in = np.array([1, 1, 1])
    omega_in = np.array([2, 1, 1])
    b = np.array(7 * [1])
    print(IK_velocity_null(q, v_in, omega_in, b))
