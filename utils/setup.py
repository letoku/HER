from datetime import datetime
import numpy as np
import torch
import pickle

import gymnasium as gym
from algorithms.td3 import TD3Agent
from algorithms.ddpg import DDPG


def her_setup():
    """parametrization of learning"""

    """parameters to specify:   """
    # TASK_NAME = 'FetchPickAndPlace-v2'
    # TASK_NAME = 'FetchReach-v2'
    TASK_NAME = 'FetchPickAndPlace-v2'
    LOAD = False
    LOAD_PATH = 'agent.pkl'
    EPISODES_NUM = 500000
    EXPLORATORY_EPISODES_NUM = 0
    EPISODES_BETWEEN_SAVE = 500
    MAX_STEPS_IN_EPISODE = 50
    MEMORY_SIZE = 1000000
    NOISE_SIGMA = 0.1
    NOISE_CLIP = 0.5
    HIDDEN_UNITS = 1024
    Q_POLYAK = 0.995
    PI_POLYAK = 0.995
    POLYAK_COEFFICIENT = 0.995
    GAMMA = 0.95
    BATCH_SIZE = 1024
    Q_LEARNING_RATE = 1e-3
    PI_LEARNING_RATE = 1e-3
    POLICY_UPDATE_DELAY = 2
    """---------------- parameters to specify  -----------------"""

    env = gym.make(TASK_NAME, max_episode_steps=MAX_STEPS_IN_EPISODE, render_mode='rgb_array')
    video_file_name = TASK_NAME + '_' + datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
    if LOAD:
        with open(LOAD_PATH, 'rb') as handle:
            agent = pickle.load(handle)

        return env, agent, video_file_name, 0

    """parameters determined by environment"""
    ACTION_DIM = env.action_space.shape[0]
    ACTION_MIN = -1.0
    ACTION_MAX = 1.0
    OBSERVATION_DIM = env.observation_space.spaces['observation'].shape[0] +\
                      env.observation_space.spaces['desired_goal'].shape[0]
    """parameters determined by environment"""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # agent = DDPG(memory_size=MEMORY_SIZE, states_dim=OBSERVATION_DIM, actions_dim=ACTION_DIM, action_min=ACTION_MIN,
    #                   action_max=ACTION_MAX, polyak=POLYAK_COEFFICIENT, gamma=GAMMA, batch_size=BATCH_SIZE,
    #                   q_learning_rate=Q_LEARNING_RATE, pi_learning_rate=PI_LEARNING_RATE, noise_sigma=NOISE_SIGMA,
    #                   network_hidden_units=HIDDEN_UNITS, device=device)

    agent = TD3Agent(memory_size=MEMORY_SIZE, states_dim=OBSERVATION_DIM, actions_dim=ACTION_DIM, action_min=ACTION_MIN,
                     action_max=ACTION_MAX, q_polyak=Q_POLYAK, policy_polyak=PI_POLYAK, gamma=GAMMA, batch_size=BATCH_SIZE,
                     q_learning_rate=Q_LEARNING_RATE, pi_learning_rate=PI_LEARNING_RATE, noise=NOISE_SIGMA,
                     target_noise=NOISE_SIGMA, q_hidden_units=HIDDEN_UNITS, policy_hidden_units=HIDDEN_UNITS,
                     device=device,
                     noise_clip=NOISE_CLIP,
                     policy_update_delay=POLICY_UPDATE_DELAY, her_clip=True)

    return env, agent, video_file_name, EPISODES_NUM, EXPLORATORY_EPISODES_NUM, EPISODES_BETWEEN_SAVE


#
# def setup():
#     """parametrization of learning"""
#
#     """parameters to specify:   """
#     TASK_NAME = 'FetchPickAndPlace-v2'
#     LOAD = False
#     LOAD_PATH = 'agent.pkl'
#     EPISODES_NUM = 10000
#     EXPLORATORY_EPISODES_NUM = 1000
#     EPISODES_BETWEEN_SAVE = 100
#     MEMORY_SIZE = 3000000
#     NOISE_SIGMA = 0.2
#     NOISE_CLIP = 0.5
#     HIDDEN_UNITS = 800
#     Q_POLYAK = 0.995
#     PI_POLYAK = 0.995
#     POLYAK_COEFFICIENT = 0.995
#     GAMMA = 0.99
#     BATCH_SIZE = 256
#     Q_LEARNING_RATE = 1e-5
#     PI_LEARNING_RATE = 1e-5
#     POLICY_UPDATE_DELAY = 2
#     """---------------- parameters to specify  -----------------"""
#
#     env = gym.make(TASK_NAME, render_mode='rgb_array')
#     x = env.action_space
#     action_spec, observation_spec = env.action_spec(), env.observation_spec()
#     video_file_name = TASK_NAME + '_' + datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
#     if LOAD:
#         with open(LOAD_PATH, 'rb') as handle:
#             agent = pickle.load(handle)
#
#         return env, agent, video_file_name, 0
#
#     """parameters determined by environment"""
#     ACTION_DIM = action_spec.shape[0]
#     ACTION_MIN = action_spec.minimum[0]
#     ACTION_MAX = action_spec.maximum[0]
#     OBSERVATION_DIM = 0
#     for _, v in observation_spec.items():
#         current_feature_dim = 1
#         for dim_i in v.shape:
#             current_feature_dim *= dim_i
#         OBSERVATION_DIM += current_feature_dim
#     """parameters determined by environment"""
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     # agent = DDPG(memory_size=MEMORY_SIZE, states_dim=OBSERVATION_DIM, actions_dim=ACTION_DIM, action_min=ACTION_MIN,
#     #                   action_max=ACTION_MAX, polyak=POLYAK_COEFFICIENT, gamma=GAMMA, batch_size=BATCH_SIZE,
#     #                   q_learning_rate=Q_LEARNING_RATE, pi_learning_rate=PI_LEARNING_RATE, noise_sigma=NOISE_SIGMA,
#     #                   network_hidden_units=HIDDEN_UNITS, device=device)
#
#     agent = TD3Agent(memory_size=MEMORY_SIZE, states_dim=OBSERVATION_DIM, actions_dim=ACTION_DIM, action_min=ACTION_MIN,
#                     action_max=ACTION_MAX, q_polyak=Q_POLYAK, policy_polyak=PI_POLYAK, gamma=GAMMA, batch_size=BATCH_SIZE,
#                      q_learning_rate=Q_LEARNING_RATE, pi_learning_rate=PI_LEARNING_RATE, noise=NOISE_SIGMA,
#                      target_noise=NOISE_SIGMA, q_hidden_units=HIDDEN_UNITS, policy_hidden_units=HIDDEN_UNITS,
#                      device=device,
#                      noise_clip=NOISE_CLIP,
#                      policy_update_delay=POLICY_UPDATE_DELAY)
#
#     return env, agent, video_file_name, EPISODES_NUM, EXPLORATORY_EPISODES_NUM, EPISODES_BETWEEN_SAVE