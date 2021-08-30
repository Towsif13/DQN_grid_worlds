from agent import Agent
import torch
from environment import Environment
import time

env = Environment()

agent = Agent(state_size=2, action_size=4, seed=0)

# load the weights from file
agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

#'C:/Users/Towsif/Desktop/New folder/DQN_grid_worlds/maze/DQN_target/checkpoint.pth'

for i in range(10):
    state = env.reset()
    env.render()
    trail = []
    for j in range(200):
        trail.append(state)
        action = agent.act(state)
        state, reward, done = env.step(action)
        env.render(list(map(tuple, trail)))
        time.sleep(0.4)
        if done:
            time.sleep(1)
            break
