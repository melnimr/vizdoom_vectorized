import time 
import argparse
import warnings 

from gymnasium.vector import AsyncVectorEnv
import gymnasium
from vizdoom import gymnasium_wrapper

warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument("--n_envs", type =int, default = 1, help ="Number of envs")
args =parser.parse_args()
n_episodes = 1000


seed = 42

envs = gymnasium.make_vec("VizdoomCorridor-v0", num_envs =args.n_envs, vectorization_mode="sync")
observation, info = envs.reset(seed =seed)
start = time.time()

for _ in range(n_episodes):
    
    actions = envs.action_space.sample()
    observations, rewards, terminations, truncations, infos = envs.step(actions)
   # if terminated or truncated:
   #    observation, info = env.reset()
print(f"{args.n_envs}  {n_episodes *args.n_envs /round(time.time() - start,1)}")
envs.close()
