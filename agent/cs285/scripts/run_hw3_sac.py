import os
import time
import yaml

from cs285.agents.soft_actor_critic import SoftActorCritic
from cs285.infrastructure.replay_buffer import ReplayBuffer
import cs285.env_configs

import os
import time

import numpy as np
import torch
from cs285.infrastructure import pytorch_util as ptu
import tqdm

from cs285.infrastructure import utils
from cs285.infrastructure.logger import Logger

from cs285.scripts.scripting_utils import make_logger, make_config



import argparse

# Much of the code in run_training_loop stays the same (hopefully)
# removed the env parts and the parts that forced discrete action spaces
def run_training_loop(config: dict, logger: Logger, args: argparse.Namespace, problem):

    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    # set up any necessary env variables without using env
    ep_len = config["ep_len"]
    batch_size = config["batch_size"] or batch_size

    # hardcode "obs_dim" and "ac_dim" for now with rl_double_integrator_problem values
    ob_shape = [4]
    ac_dim = 2
    agent = SoftActorCritic(
        ob_shape,
        ac_dim,
        **config["agent_kwargs"],
    )

    replay_buffer = ReplayBuffer(config["replay_buffer_capacity"])
    # initial_observation = np.array([0, 0]) # hardcode for now, generate through rl_double_integrator_problem later

    # replace env.reset with initial_observation
    observation = utils.reset_env()
    problem.update(initial_state=(5 * observation[:2]), horizon=int(250 + 150 * observation[2]), reset_prop=observation[3])

    for step in tqdm.trange(config["total_steps"], dynamic_ncols=True):

        # TODO: Not sure if we should keep a random step section
        if step < config["random_steps"]:
            action = utils.sample_actions()
        else:
            # TODO(student): Select an action
            action = agent.get_action(observation)

        # Step the environment and add the data to the replay buffer
        # TODO: need to replace env.step with RHDDP step
        next_observation, reward, done = utils.step(action, problem)
        replay_buffer.insert(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            done=done,
        )

        # TODO: should be able to simplify this, as done is always True
        logger.log_scalar(reward, "train_return", step)
        logger.log_scalar(1, "train_ep_len", step)
        observation = utils.reset_env()

        # Train the agent
        if step >= config["training_starts"]:
            # TODO(student): Sample a batch of config["batch_size"] transitions from the replay buffer
            batch = replay_buffer.sample(config["batch_size"])
            update_info = agent.update(**batch, step=step)

            # Logging
            update_info["actor_lr"] = agent.actor_lr_scheduler.get_last_lr()[0]
            update_info["critic_lr"] = agent.critic_lr_scheduler.get_last_lr()[0]

            if step % args.log_interval == 0:
                for k, v in update_info.items():
                    logger.log_scalar(v, k, step)
                    logger.log_scalars
                logger.flush()

        # Run evaluation
        # TODO: not exactly sure how we deal with this section that samples trajectories, maybe we should just do logging differently
        if step % args.eval_interval == 0:
            trajectories = utils.sample_n_trajectories(
                policy=agent,
                ntraj=args.num_eval_trajectories,
                max_length=ep_len,
                problem=problem
            )
            returns = [t["episode_statistics"]["r"] for t in trajectories]
            ep_lens = [t["episode_statistics"]["l"] for t in trajectories]

            logger.log_scalar(np.mean(returns), "eval_return", step)
            logger.log_scalar(np.mean(ep_lens), "eval_ep_len", step)

            if len(returns) > 1:
                logger.log_scalar(np.std(returns), "eval/return_std", step)
                logger.log_scalar(np.max(returns), "eval/return_max", step)
                logger.log_scalar(np.min(returns), "eval/return_min", step)
                logger.log_scalar(np.std(ep_lens), "eval/ep_len_std", step)
                logger.log_scalar(np.max(ep_lens), "eval/ep_len_max", step)
                logger.log_scalar(np.min(ep_lens), "eval/ep_len_min", step)

            # if args.num_render_trajectories > 0:
            #     video_trajectories = utils.sample_n_trajectories(
            #         render_env,
            #         agent,
            #         args.num_render_trajectories,
            #         ep_len,
            #         render=True,
            #     )

            #     logger.log_paths_as_videos(
            #         video_trajectories,
            #         step,
            #         fps=fps,
            #         max_videos_to_save=args.num_render_trajectories,
            #         video_title="eval_rollouts",
            #     )

#region oldcode
"""
def run_training_loop(config: dict, logger: Logger, args: argparse.Namespace):
    # set random seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    # make the gym environment
    env = config["make_env"]()
    eval_env = config["make_env"]()
    render_env = config["make_env"](render=True)

    ep_len = config["ep_len"] or env.spec.max_episode_steps
    batch_size = config["batch_size"] or batch_size

    discrete = isinstance(env.action_space, gym.spaces.Discrete)
    assert (
        not discrete
    ), "Our actor-critic implementation only supports continuous action spaces. (This isn't a fundamental limitation, just a current implementation decision.)"

    ob_shape = env.observation_space.shape
    ac_dim = env.action_space.shape[0]

    # simulation timestep, will be used for video saving
    if "model" in dir(env):
        fps = 1 / env.model.opt.timestep
    else:
        fps = env.env.metadata["render_fps"]

    # initialize agent
    agent = SoftActorCritic(
        ob_shape,
        ac_dim,
        **config["agent_kwargs"],
    )

    replay_buffer = ReplayBuffer(config["replay_buffer_capacity"])

    observation = env.reset()

    for step in tqdm.trange(config["total_steps"], dynamic_ncols=True):
        if step < config["random_steps"]:
            action = env.action_space.sample()
        else:
            # TODO(student): Select an action
            action = agent.get_action(observation)

        # Step the environment and add the data to the replay buffer
        next_observation, reward, done, info = env.step(action)
        replay_buffer.insert(
            observation=observation,
            action=action,
            reward=reward,
            next_observation=next_observation,
            done=done and not info.get("TimeLimit.truncated", False),
        )

        if done:
            logger.log_scalar(info["episode"]["r"], "train_return", step)
            logger.log_scalar(info["episode"]["l"], "train_ep_len", step)
            observation = env.reset()
        else:
            observation = next_observation

        # Train the agent
        if step >= config["training_starts"]:
            # TODO(student): Sample a batch of config["batch_size"] transitions from the replay buffer
            batch = replay_buffer.sample(config["batch_size"])
            update_info = agent.update(**batch, step=step)

            # Logging
            update_info["actor_lr"] = agent.actor_lr_scheduler.get_last_lr()[0]
            update_info["critic_lr"] = agent.critic_lr_scheduler.get_last_lr()[0]

            if step % args.log_interval == 0:
                for k, v in update_info.items():
                    logger.log_scalar(v, k, step)
                    logger.log_scalars
                logger.flush()

        # Run evaluation
        if step % args.eval_interval == 0:
            trajectories = utils.sample_n_trajectories(
                eval_env,
                policy=agent,
                ntraj=args.num_eval_trajectories,
                max_length=ep_len,
            )
            returns = [t["episode_statistics"]["r"] for t in trajectories]
            ep_lens = [t["episode_statistics"]["l"] for t in trajectories]

            logger.log_scalar(np.mean(returns), "eval_return", step)
            logger.log_scalar(np.mean(ep_lens), "eval_ep_len", step)

            if len(returns) > 1:
                logger.log_scalar(np.std(returns), "eval/return_std", step)
                logger.log_scalar(np.max(returns), "eval/return_max", step)
                logger.log_scalar(np.min(returns), "eval/return_min", step)
                logger.log_scalar(np.std(ep_lens), "eval/ep_len_std", step)
                logger.log_scalar(np.max(ep_lens), "eval/ep_len_max", step)
                logger.log_scalar(np.min(ep_lens), "eval/ep_len_min", step)

            if args.num_render_trajectories > 0:
                video_trajectories = utils.sample_n_trajectories(
                    render_env,
                    agent,
                    args.num_render_trajectories,
                    ep_len,
                    render=True,
                )

                logger.log_paths_as_videos(
                    video_trajectories,
                    step,
                    fps=fps,
                    max_videos_to_save=args.num_render_trajectories,
                    video_title="eval_rollouts",
                )
"""
#endregion oldcode
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", "-cfg", type=str, required=True)

    parser.add_argument("--eval_interval", "-ei", type=int, default=5000)
    parser.add_argument("--num_eval_trajectories", "-neval", type=int, default=10)
    parser.add_argument("--num_render_trajectories", "-nvid", type=int, default=0)

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-g", default=0)
    parser.add_argument("--log_interval", type=int, default=1000)

    args = parser.parse_args()

    # create directory for logging
    logdir_prefix = "fp_"  # keep for autograder

    config = make_config(args.config_file)
    logger = make_logger(logdir_prefix, config)

    problem = utils.setup_problem()
    
    
    run_training_loop(config, logger, args, problem)


if __name__ == "__main__":
    main()
