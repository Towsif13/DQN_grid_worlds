from agent import Agent
import torch
from environment import Environment
import time

env = Environment()

agent = Agent(state_size=2, action_size=4, seed=0)

# load the weights from file
agent.qnetwork_local.load_state_dict(
    torch.load('checkpoint.pth'))

for i in range(1):
    state = env.reset()
    trail = []
    for j in range(200):

        action = agent.act(state)
        state, reward, done = env.step(action)
        trail.append(state)
        env.render()
        # time.sleep(0.4)
        if done:
            time.sleep(1)

            break
    print(trail)
