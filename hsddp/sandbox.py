import numpy as np

num_test_points=15

dset = [np.array([-0.05, -0.5]), np.array([-0.05, 0.5]), np.array([0.05, -0.5]), np.array([0.05, 0.5])]
# d0_min = -200
# d0_max = -100
# d1_min = 1
# d1_max = 3

d_min = np.min(dset, axis=0)
d_max = np.max(dset, axis=0)

num_d0_points = num_test_points
num_d1_points = num_test_points

d0_values = np.linspace(d_min[0], d_max[0], num_d0_points)
d1_values = np.linspace(d_min[1], d_max[1], num_d1_points)

distGrid = np.array([(d0, d1) for d0 in d0_values for d1 in d1_values])

pass
