from gym.envs.registration import register

register(
    id='nav2dVeryEasy-v0',
    entry_point='gym_nav2d.envs:Nav2dVeryEasyEnv',
)

register(
    id='nav2dEasy-v0',
    entry_point='gym_nav2d.envs:Nav2dEasyEnv',
)

register(
    id='nav2dHard-v0',
    entry_point='gym_nav2d.envs:Nav2dHardEnv',
)

register(
    id='nav2dVeryHard-v0',
    entry_point='gym_nav2d.envs:Nav2dVeryHardEnv',
)

register(
    id='nav2dmdpgoal-v0',
    entry_point='gym_nav2d.envs:Nav2dMDPGoal',
)

register(
    id='nav2dpomdpgoal-v0',
    entry_point='gym_nav2d.envs:Nav2dPOMDPGoal',
)
