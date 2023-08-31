import fedscale.cloud.aggregation.RL_singleQ_individual_optimization as RL
def load_Q():
    rl_agent = RL()
    Q = rl_agent.load_Q('/content/')
    print(Q)

load_Q()