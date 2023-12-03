class Trajectory:
    def __init__(self, x_traj, u_traj, dist, cost):
        self.x_traj = x_traj
        self.u_traj = u_traj
        self.cost = cost
        self.dist = dist