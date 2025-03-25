def set_time_dependent_hparams(episode: int, EPISODES_NUM: int, EXPLORATORY_EPISODES_NUM: int, max_noise: float) ->\
        (bool, float):
    exploratory_phase = False if episode >= EXPLORATORY_EPISODES_NUM else True

    if exploratory_phase:
        noise = max_noise
    else:
        noise = max_noise * (1 - (episode - EXPLORATORY_EPISODES_NUM) / (EPISODES_NUM - EXPLORATORY_EPISODES_NUM))

    return exploratory_phase, noise