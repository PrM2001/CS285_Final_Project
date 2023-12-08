import numpy as np
import matplotlib.pyplot as plt

def plot_compass_gait_traj(x_traj, u_traj, cost):
    traj_length = x_traj.shape[1]
    ax_x = plt.subplot(421)
    ax_x.plot(np.arange(traj_length), x_traj[0, :].reshape(-1))
    ax_x.set_ylabel("p_x")
    ax_x.grid()

    ax_y = plt.subplot(423)
    ax_y.plot(np.arange(traj_length), x_traj[1, :].reshape(-1))
    ax_y.set_ylabel("p_y")
    ax_y.grid()

    ax3 = plt.subplot(425)
    ax3.plot(np.arange(traj_length), x_traj[2, :].reshape(-1))
    ax3.set_ylabel("q1")
    ax3.grid()

    ax4 = plt.subplot(427)
    ax4.plot(np.arange(traj_length), x_traj[3, :].reshape(-1))
    ax4.set_ylabel("q2")
    ax4.grid()

    ax_dx = plt.subplot(422)
    ax_dx.plot(np.arange(traj_length), x_traj[4, :].reshape(-1))
    ax_dx.set_ylabel("v_x")
    ax_dx.grid()

    ax_dy = plt.subplot(424)
    ax_dy.plot(np.arange(traj_length), x_traj[5, :].reshape(-1))
    ax_dy.set_ylabel("v_y")
    ax_dy.grid()

    ax7 = plt.subplot(426)
    ax7.plot(np.arange(traj_length), x_traj[6, :].reshape(-1))
    ax7.set_ylabel("dq1")
    ax7.grid()

    ax8 = plt.subplot(428)
    ax8.plot(np.arange(traj_length), x_traj[7, :].reshape(-1))
    ax8.set_ylabel("dq2")
    ax8.grid()
    plt.show()

    def swing_foot_pos(q):
        x = q[0] + np.sin(q[2] + q[3])
        y = q[1] - np.cos(q[2] + q[3])
        return np.array([x, y])

    pos_swing_foot = np.zeros((2, traj_length))
    for i in range(traj_length):
        pos_swing_foot[:, i] = swing_foot_pos(x_traj[:, i])    

    plt.plot(pos_swing_foot[0, :], pos_swing_foot[1, :])
    plt.plot(x_traj[0, :], x_traj[1, :])
    plt.grid()
    plt.ylabel("swing foot pos")
    plt.show()

    plt.plot(np.arange(u_traj.shape[1]), u_traj.reshape(-1))
    plt.grid()
    plt.ylabel("u")
    plt.show()

    # cost_test = np.zeros(traj_length)
    # for i in range(traj_length):
    #     cost_test[i] = cost.l(x_traj[:, i], u_traj[:, i])
    # plt.plot(np.arange(traj_length), cost_test)
    # plt.show()

    plt.show()