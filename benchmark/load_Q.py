import fedscale.cloud.aggregation.RL_singleQ as RL
def load_Q():
    rl_agent = RL()
    Q = rl_agent.load_Q('/content/')
    print(Q)

load_Q()