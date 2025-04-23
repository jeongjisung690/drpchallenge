import torch as th


from functools import partial
from .epymarl.src.components.episode_buffer import EpisodeBatch
from .epymarl.src.controllers.basic_controller import BasicMAC
from .epymarl.src.components.transforms import OneHot

def policy(n_obs,env):
    env_info = env.get_env_info()

    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "avail_actions": {
            "vshape": (env_info["n_actions"],),
            "group": "agents",
            "dtype": th.int,
        },
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {"agents": env_info["n_actions"]}
    preprocess = {"actions": ("actions_onehot", [OneHot(out_dim=env_info["n_actions"])])}

    new_batch = partial(EpisodeBatch, scheme, groups, batch_size=32, episode_limit=100 + 1,
                                 preprocess=preprocess, device="cuda")
    batch = new_batch()


    pre_transition_data = {
    "state": [env.get_state()],
    "avail_actions": [env.get_avail_actions()],
    "obs": [env.get_obs()]
    }
    batch.update(pre_transition_data, ts=env.step_account)

    actions = 

