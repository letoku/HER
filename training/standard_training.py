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


def train(env, agent, video_file_name, EPISODES_NUM, EXPLORATORY_EPISODES_NUM, EPISODES_BETWEEN_SAVE):
    episodes_scores = []
    max_noise = 0.4
    action_sel_t, env_step_t, update_t = [], [], []

    for episode in tqdm(range(EPISODES_NUM)):
        state, _, _ = env.reset()
        exploratory_phase, noise = set_time_dependent_hparams(episode, EPISODES_NUM=EPISODES_NUM,
                                                              EXPLORATORY_EPISODES_NUM=EXPLORATORY_EPISODES_NUM,
                                                              max_noise=max_noise)
        terminated, episode_score, steps_performed = False, 0, 0

        while not terminated:
            if exploratory_phase:
                action = env.random_action()
            else:
                action = agent.get_action(state=state, noise=noise)

            next_state, reward, terminated = env.step(action)
            agent.push(Transition(state=state, action=action, next_state=next_state, reward=reward,
                                  terminated=terminated))
            state = next_state

            if not exploratory_phase:
                agent.update()
            steps_performed += 1
            episode_score += reward

        episodes_scores.append(episode_score)
        if episode_score > 50:
            with open(f'td3_{video_file_name}_good_score.pkl', 'wb') as handle:
                pickle.dump(agent, handle, protocol=pickle.HIGHEST_PROTOCOL)

        log_run(episodes_scores=episodes_scores, env=env, agent=agent, video_file_name=video_file_name,
                episode=episode, checkpoint=EPISODES_BETWEEN_SAVE)
