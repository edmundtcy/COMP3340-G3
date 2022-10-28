import torch as th
import gym
from tqdm import tqdm, trange
from sb3_contrib import TRPO
import matplotlib.pyplot as plt

num_features = 128
# Custom actor (pi) and value function (vf) networks
policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=[dict(pi=[num_features, num_features, num_features], vf=[num_features, num_features, num_features])])
                                            
env = gym.make('LunarLanderContinuous-v2')

model = TRPO('MlpPolicy', env, policy_kwargs=policy_kwargs, gamma=0.99, target_kl=0.01, batch_size=64,verbose=1)

episodes = 30
reward_per_episode = []

for episode in trange(episodes):
    model = model.learn(total_timesteps=500, log_interval=4)
    state = env.reset()
    total_reward, total_step = 0, 0
    while True:
        env.render()
        action, _state = model.predict(state)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        total_reward += reward
        total_step += 1
        if done:
            env.reset()
            reward_per_episode.append(total_reward)
            break
    # print("_______________________________________________")
    # print("episode :", episode)
    # print("reward: ",reward)
    # print("total reward: ",total_reward)
    # print("total steps: ",total_step)
env.close()
plt.plot(reward_per_episode)
plt.title("Rewards per episode")
plt.show()

