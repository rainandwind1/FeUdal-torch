import gym
from param import *
from utils import *
from model import *
from wrapper import *
# import os
# os.add_dll_directory("D:/Anaconda/lib/site-packages/atari_py/ale_interface/ale_c.dll")


def run():
    state_size = 4
    action_size = 2
    d_dim = 32
    k_dim = 16
    c = 10
    alpha = 0.6
    # self.state_size, self.action_size, self.d_dim, self.k_dim, self.lr, self.c, self.alpha, self.device = args
    model = FeUdal(args = (4, 2, 32, 16, 1e-3, C, 0.6, DEVICE)).to(DEVICE)
    
    # env_id = "PongNoFrameskip-v4"
    # env = make_atari(env_id)
    # env = wrap_deepmind(env)
    # env = wrap_pytorch(env)
    env = gym.make("CartPole-v1")


    total_step = 0
    for epi_i in range(MAX_EPISODES):
        score = 0.
        s = env.reset()
        hidden_m = torch.randn(1, 1, d_dim).to(DEVICE)
        hidden_w = torch.randn(1, 1, action_size * k_dim).to(DEVICE)
        for step_i in range(MAX_STEPS):
            action, hidden_m, hidden_w = model.get_action(torch.FloatTensor(s).to(DEVICE), hidden_m, hidden_w)
            s_next, r, done, info = env.step(action)
            # mem
            model.ep_action_ls.append([action])
            model.ep_state_ls.append(s)
            model.ep_reward_ls.append([r])

            s = s_next
            score += 1
            if done:
                break
        model.train()
        print("Episode: {}  score: {}".format(epi_i + 1, score))

def test_env():
    env = gym.make('Alien-ram-v0')
    for i_episode in range(20):
        observation = env.reset()
        for t in range(100):
            env.render()
            print(observation)
            action = env.action_space.sample()
            observation, reward, done, info = env.step(action)
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break
    env.close()


if __name__ == "__main__":
    # test_env()
    run()
