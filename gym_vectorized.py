from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor, VecTransposeImage
from stable_baselines3.common.env_util import make_vec_env


from gymnasium.vector import AsyncVectorEnv
import gymnasium
from vizdoom import gymnasium_wrapper


seed = 42
# def make_env(env_id, rank, seed=0):
#     def _init():
#         env = gymnasium.make(env_id)
#         #env.seed(seed + rank)
#         return env
#     return _init

# # envs = gymnasium.make_vec("VizdoomCorridor-v0", num_envs=5, vectorization_mode="async")
# num_envs = 5
env_id = "VizdoomCorridor-v0"

env = gymnasium.make(env_id)
# env_list = [make_env(env_id, i) for i in range(5)]
# envs = AsyncVectorEnv(env_list)

model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)
model.save("vizdoom_vectorized")

#monitor rewards
envs = VecMonitor(env)

observation, info = envs.reset()
for _ in range(1000):
    action, _states = model.predict(obs)
    observation, reward, terminated, truncated, info = envs.step(action)
   # if terminated or truncated:
   #    observation, info = env.reset()

envs.close()