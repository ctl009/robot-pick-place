import numpy as np


def calcAngDiff(R_des, R_curr):
    """
    Helper function for the End Effector Orientation Task. Computes the axis of rotation
    from the current orientation to the target orientation

    This data can also be interpreted as an end effector velocity which will
    bring the end effector closer to the target orientation.

    INPUTS:
    R_des - 3x3 numpy array representing the desired orientation from
    end effector to world
    R_curr - 3x3 numpy array representing the "current" end effector orientation

    OUTPUTS:
    omega - 0x3 a 3-element numpy array containing the axis of the rotation from
    the current frame to the end effector frame. The magnitude of this vector
    must be sin(angle), where angle is the angle of rotation around this axis
    """
    omega = np.zeros(3)

    ## STUDENT CODE STARTS HERE
    R = np.matmul(np.transpose(R_des), R_curr).T
    # theta = np.arccos((np.trace(R) - 1) / 2)
    Skew = 0.5 * (R - R.T)
    omega = np.array([Skew[2, 1], Skew[0, 2], Skew[1, 0]])
    return np.matmul(R_curr, omega)


if __name__ == "__main__":
    R_desired = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    R_current = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])

    omega = calcAngDiff(R_desired, R_current)
    print(np.sqrt(omega[0]**2 + omega[1]**2 +omega[2]**2))
    print(omega)
