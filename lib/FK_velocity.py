import numpy as np
from lib.calcJacobian import calcJacobian


def FK_velocity(q_in, dq):
    """
    :param q_in: 1 x 7 vector corresponding to the robot's current configuration.
    :param dq: 1 x 7 vector corresponding to the joint velocities.
    :return:
    velocity - 6 x 1 vector corresponding to the end effector velocities.
    """

    ## STUDENT CODE GOES HERE

    # velocity = np.zeros((6, 1))
    J = calcJacobian(q_in)
    velocity = (J @ dq).reshape((6, 1))
    # print(velocity.shape)
    return velocity


if __name__ == "__main__":
    dq = np.array([1, 0, 0, 0, 0, 0, 0])
    q = np.array([0, 0, 0, -np.pi / 2, 0, np.pi / 2, np.pi / 4])
    print(np.round(FK_velocity(q, dq), 3))
