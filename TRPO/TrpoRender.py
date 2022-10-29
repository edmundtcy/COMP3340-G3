import torch as th
import gym
from tqdm import tqdm, trange
from sb3_contrib import TRPO
import matplotlib.pyplot as plt

if(th.cuda.is_available()): 
    device = th.device('cuda:0') 
    th.cuda.empty_cache()
    print("Device set to : " + str(th.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

num_features = 128 # number of neurals per layers
num_layers = 3
episodes = 1000
# Custom actor (pi) and value function (vf) networks
# Custom MLP policy of three layers of size 128 each
policy_kwargs = dict(activation_fn=th.nn.ReLU,
                     net_arch=[dict(pi=[num_features]*num_layers, vf=[num_features]*num_layers)])
                                            
env = gym.make('LunarLanderContinuous-v2')

model = TRPO('MlpPolicy', env, policy_kwargs=policy_kwargs, gamma=0.99, target_kl=0.01, batch_size=64,verbose=1)
model_name = "trpo_{}F_{}L_{}Episo_{}BatchS_{}γ_{}LearnR_{}TargetKL".format(num_features,num_layers, episodes,model.batch_size,model.gamma,model.learning_rate,model.target_kl)

reward_per_episode = []

for episode in trange(episodes):
    model.learn(total_timesteps=1000, log_interval=4)
    model.save(model_name)
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

env.close()
plt.plot(reward_per_episode)
plt.title("Rewards per episode")
plt.show()

del model
model = TRPO.load("trpo_lunar")
while True:
    env.render()
    action, _state = model.predict(state)
    next_state, reward, done, _ = env.step(action)
    state = next_state
    if done:
        env.reset()


