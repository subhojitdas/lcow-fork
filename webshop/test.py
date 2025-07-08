import gym
from web_agent_site.envs import WebAgentTextEnv
from web_agent_site.utils import DEBUG_PROD_SIZE

# intialize environment
env = gym.make('WebAgentTextEnv-v0', observation_mode='text_rich', num_products=DEBUG_PROD_SIZE)

# reset environment (each task idx corresponds to different human instruction)
task_idx = 0 # (0 ~ 6510)
obs, info = env.reset(session=task_idx)
print(obs)
action = 'search[iphone 13 pro]'
# reward is given at the end of the episode (when you click[buy now]), based on the task information.
obs, reward, done, info = env.step(action)
print(obs)