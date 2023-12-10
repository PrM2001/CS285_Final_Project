from collections import OrderedDict
import numpy as np
import copy
from cs285.networks.mlp_policy import MLPPolicy
import cv2
from cs285.infrastructure import pytorch_util as ptu
from typing import Dict, Tuple, List

from rhddp.problem import Problem
from rhddp.cost import SymbolicCost
from rhddp.dynamics import SymbolicDynamics
from rhddp.rhddp import RHDDP

############################################
############################################


def sample_trajectory(
    policy: MLPPolicy, max_length: int, render: bool = False, problem=None
) -> Dict[str, np.ndarray]:
    """Sample a rollout in the environment from a policy."""
    ob = reset_env()
    obs, acs, rewards, next_obs, terminals, image_obs = [], [], [], [], [], []
    steps = 0

    while True:
        # TODO use the most recent ob to decide what to do
        ac = policy.get_action(ob).reshape((2,2))

        # TODO: take that action and get reward and next ob
        next_ob, rew, done = step(ac, problem)

        # TODO rollout can end due to done, or due to max_length
        steps += 1
        rollout_done = done or steps > max_length  # HINT: this is either 0 or 1

        # record result of taking that action
        obs.append(ob)
        acs.append(ac)
        rewards.append(rew)
        next_obs.append(next_ob)
        terminals.append(rollout_done)

        ob = next_ob  # jump to next timestep

        # end the rollout if the rollout ended
        if rollout_done:
            break

    episode_statistics = {"l": steps, "r": np.sum(rewards)}
    # if "episode" in info:
    #     episode_statistics.update(info["episode"])


    return {
        "observation": np.array(obs, dtype=np.float32),
        "image_obs": np.array(image_obs, dtype=np.uint8),
        "reward": np.array(rewards, dtype=np.float32),
        "action": np.array(acs, dtype=np.float32),
        "next_observation": np.array(next_obs, dtype=np.float32),
        "terminal": np.array(terminals, dtype=np.float32),
        "episode_statistics": episode_statistics,
    }


def sample_n_trajectories(
    policy: MLPPolicy, ntraj: int, max_length: int, render: bool = False, problem=None
):
    """Collect ntraj rollouts."""
    trajs = []
    for _ in range(ntraj):
        # collect rollout
        traj = sample_trajectory(policy, max_length, render, problem)
        trajs.append(traj)
    return trajs


def compute_metrics(trajs, eval_trajs):
    """Compute metrics for logging."""

    # returns, for logging
    train_returns = [traj["reward"].sum() for traj in trajs]
    eval_returns = [eval_traj["reward"].sum() for eval_traj in eval_trajs]

    # episode lengths, for logging
    train_ep_lens = [len(traj["reward"]) for traj in trajs]
    eval_ep_lens = [len(eval_traj["reward"]) for eval_traj in eval_trajs]

    # decide what to log
    logs = OrderedDict()
    logs["Eval_AverageReturn"] = np.mean(eval_returns)
    logs["Eval_StdReturn"] = np.std(eval_returns)
    logs["Eval_MaxReturn"] = np.max(eval_returns)
    logs["Eval_MinReturn"] = np.min(eval_returns)
    logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)

    logs["Train_AverageReturn"] = np.mean(train_returns)
    logs["Train_StdReturn"] = np.std(train_returns)
    logs["Train_MaxReturn"] = np.max(train_returns)
    logs["Train_MinReturn"] = np.min(train_returns)
    logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

    return logs


def convert_listofrollouts(trajs):
    """
    Take a list of rollout dictionaries and return separate arrays, where each array is a concatenation of that array
    from across the rollouts.
    """
    observations = np.concatenate([traj["observation"] for traj in trajs])
    actions = np.concatenate([traj["action"] for traj in trajs])
    next_observations = np.concatenate([traj["next_observation"] for traj in trajs])
    terminals = np.concatenate([traj["terminal"] for traj in trajs])
    concatenated_rewards = np.concatenate([traj["reward"] for traj in trajs])
    unconcatenated_rewards = [traj["reward"] for traj in trajs]
    return (
        observations,
        actions,
        next_observations,
        terminals,
        concatenated_rewards,
        unconcatenated_rewards,
    )


def get_traj_length(traj):
    return len(traj["reward"])



def setup_problem():
    dynamics = SymbolicDynamics(name="double_integrator_dynamics",
                                                num_states=2,
                                                num_inputs=1,
                                                num_disturbances=1,
                                                codegen_dir='rhddp.gen')

    cost = SymbolicCost(name="double_integrator_cost",
                                        num_states=2,
                                        num_inputs=1,
                                        codegen_dir='rhddp.gen')

    d_nom = 0.0
    hyperparams = {"max_iters": 5, 
                "conv_criterion": 8} 

    settings = {"initial_state": np.array([0, 0]),
                "horizon": 1000,
                "d_nom": d_nom,
                "reset_prop": 0.8,
                }

    problem = Problem(name="double_integrator_baseline",
                    dyn=dynamics,
                    cost=cost,
                    settings=settings,
                    hyperparams=hyperparams)
    return problem



def reset_env():
    # try 0->1 or -1->1
    initial_state = np.random.uniform(-1, 1, 4)
    return initial_state

def sample_actions():
    actions = np.random.normal(0, 1, 4)
    return actions


def step(action, problem, k=4):
    vanilla_controller = RHDDP(problem, action=None)
    vanilla_solution = vanilla_controller.solve()

    rl_controller = RHDDP(problem, action=action)
    rl_solution = rl_controller.solve()

    vanilla_x_traj = vanilla_solution.get("x_traj")
    vanilla_u_traj = vanilla_solution.get("u_traj")
    vanilla_K_traj = vanilla_solution.get("K_traj")

    rl_x_traj = rl_solution.get("x_traj")
    rl_u_traj = rl_solution.get("u_traj")
    rl_K_traj = rl_solution.get("K_traj")

    reward = 0
    for _ in range(k):
        #TODO: Draw disturbance:
        disturbance = np.array([0.0])
        c_nom = vanilla_controller.rollout_cost(vanilla_x_traj, vanilla_u_traj, vanilla_K_traj, disturbance)
        c_rl = rl_controller.rollout_cost(rl_x_traj, rl_u_traj, rl_K_traj, disturbance)
        reward += (c_nom - c_rl)/c_nom 
        
        #TODO: create evaluation function for each, on the disturbance
    reward /= k
    # need to roll out and compute cost over disturbances.
    return np.zeros(4), reward, np.dtype('int32').type(1)
