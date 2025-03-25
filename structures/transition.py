from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'terminated'))
TempHerTransition = namedtuple('TempHerTransition',
                           ('obs', 'action', 'next_obs', 'info', 'terminated', 'truncated', 'achieved_goal',
                            'desired_goal'))
