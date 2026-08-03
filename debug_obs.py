import gymnasium as gym
import lbforaging

env = gym.make("Foraging-6x6-2p-1f-v3")

obs, info = env.reset()

print(vars(env.unwrapped))