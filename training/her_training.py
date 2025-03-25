import random

import numpy as np
from datetime import datetime
import pickle
from tqdm import tqdm

from utils.visualizations_helpers import *
from utils.setup import *
from structures.transition import *
from structures.replay_memory import ReplayMemory
from utils.run_logging import log_run, evaluate
from training.utils import set_time_dependent_hparams
import time
from typing import List


def add_episode_to_memory(env, agent, transitions_from_episode: List[TempHerTransition], strategy: str = 'final'):
    for i, transition in enumerate(transitions_from_episode):
        agent.push(ReplayMemory.create_HER_transition(env, transition, transition.desired_goal))
        # additional HER transition
        if strategy == 'final':
            desired_goal = transitions_from_episode[-1].achieved_goal
        else:
            # TO DO: add another strategies
            desired_goal = transitions_from_episode[-1].achieved_goal
        agent.push(ReplayMemory.create_HER_transition(env, transition, desired_goal))


def her_train(env, agent, video_file_name, total_epochs, EXPLORATORY_EPISODES_NUM, EPISODES_BETWEEN_SAVE):
    episodes_scores = []
    max_noise = 0.4
    total_cycles = 50
    batches_updates = 40
    episodes_in_cycle = 16

    # for checking
    obs_max, obs_min = np.matrix(np.ones((25, )) * (-np.inf)), np.matrix(np.ones((25, )) * np.inf)

    for epoch in tqdm(range(total_epochs)):
        for cycle in range(total_cycles):
            for episode in range(episodes_in_cycle):
                obs, info = env.reset()
                exploratory_phase, noise = set_time_dependent_hparams(epoch, EPISODES_NUM=total_epochs,
                                                                      EXPLORATORY_EPISODES_NUM=EXPLORATORY_EPISODES_NUM,
                                                                      max_noise=max_noise)
                terminated, truncated, episode_score, steps_performed = False, False, 0, 0
                transitions = []

                while not terminated and not truncated:
                    explore = random.uniform(0, 1)  # HER exploration
                    if explore < 0.2:
                        action = env.action_space.sample()
                    else:
                        action = agent.get_action(state=np.concatenate((obs['observation'].reshape(1, -1),
                                                                        obs['desired_goal'].reshape(1, -1)), axis=1),
                                                  noise=noise)

                    obs_min, obs_max = np.minimum(obs_min, obs['observation']), np.maximum(obs_max, obs['observation'])
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    transitions.append(TempHerTransition(obs=obs['observation'],
                                                         action=action, next_obs=next_obs['observation'],
                                                         info=info,
                                                         terminated=terminated,
                                                         truncated=truncated,
                                                         desired_goal=obs['desired_goal'],
                                                         achieved_goal=next_obs['achieved_goal']))
                    obs = next_obs
                    steps_performed += 1
                    episode_score += reward

                add_episode_to_memory(env, agent, transitions_from_episode=transitions)
                episodes_scores.append(episode_score)

            for batch in range(batches_updates):
                agent.update()

            if agent.standardizing:
                agent.states_standardizer.update_params()
                agent.actions_standardizer.update_params()

        log_run(episodes_scores=episodes_scores, env=env, agent=agent, video_file_name=video_file_name,
                episode=epoch, checkpoint=1, env_type='multigoal')