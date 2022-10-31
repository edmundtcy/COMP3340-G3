import torch as th
import gym
from tqdm import trange
from sb3_contrib import TRPO
import matplotlib.pyplot as plt

if(th.cuda.is_available()): 
    device = th.device('cuda:0') 
    th.cuda.empty_cache()
    print("Device set to : " + str(th.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")

num_features = 128 # number of neurals per layers
num_layers = 2
episodes = 1000
act_f = th.nn.ReLU
# Custom actor (pi) and value function (vf) networks
# Custom MLP policy of three layers of size 128 each
policy_kwargs = dict(activation_fn=act_f,
                     net_arch=[dict(pi=[num_features]*num_layers, vf=[num_features]*num_layers)])
                                            
env = gym.make('LunarLanderContinuous-v2')

model = TRPO('MlpPolicy', env, policy_kwargs=policy_kwargs, gamma=0.99, target_kl=0.01, batch_size=64,verbose=1)
model_name = "trpo_{}F_{}L_{}Episo_{}BatchS_{}γ_{}LearnR_{}TargetKL_{}".format(num_features,num_layers, episodes,model.batch_size,model.gamma,model.learning_rate,model.target_kl,act_f.__name__)

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
plt.axhline(y = 200, color = 'g', linestyle = '-')
plt.axhline(y = 0, color = 'r', linestyle = '-')
plt.plot(reward_per_episode)
plt.title("{}, Rewards per episode".format(model_name))
plt.show()