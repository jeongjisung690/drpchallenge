import os
from .run_policy import PolicyRunner

TEAM_NAME = "HOGE"
runner = None


def get_model_path(env):
    base_dir = os.path.dirname(os.path.abspath(__file__))
    filename = f"{env.map_name}_{env.agent_num}.th"
    path = os.path.join(base_dir, "models", filename)

    return path


def policy(obs, env):
    global runner

    if runner is None:
        runner = PolicyRunner(
            model_path=get_model_path(env),
            #model_path="agent.th",
            input_shape=len(obs[0]),
            n_actions=env.n_actions,
            agent_num=env.agent_num
        )

    
    actions = []
    for agi in range(env.agent_num):
        _, avail_actions = env.get_avail_agent_actions(agi, env.n_actions)
        action = runner.get_action(agi, obs[agi], avail_actions)
        actions.append(action)

    # safety control
    for i in range(env.agent_num):
        if env.current_goal[i] != None:
            continue
        for j in range(env.agent_num):
            if i == j:
                continue
            if actions[i] == actions[j]:
                actions[i] = env.current_start[i]
                break
            elif actions[i] == env.current_start[j] and actions[j] == env.current_start[i]:
                actions[i] = env.current_start[i]
                break    

    return actions