import os
import sys

from .multiagentenv import MultiAgentEnv
from .gymma import GymmaWrapper

# Disabled because SMAClite is not installed and is not needed for this project.
# from .smaclite_wrapper import SMACliteWrapper


if sys.platform == "linux":
    os.environ.setdefault(
        "SC2PATH", os.path.join(os.getcwd(), "3rdparty", "StarCraftII")
    )


def __check_and_prepare_smac_kwargs(kwargs):
    assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
    assert kwargs[
        "common_reward"
    ], (
        "SMAC only supports common reward. "
        "Please set `common_reward=True` or choose a different environment."
    )
    del kwargs["common_reward"]
    del kwargs["reward_scalarisation"]
    assert "map_name" in kwargs, "Please specify the map_name in the env_args"
    return kwargs


# Disabled because SMAClite is not installed.
#
# def smaclite_fn(**kwargs) -> MultiAgentEnv:
#     kwargs = __check_and_prepare_smac_kwargs(kwargs)
#     return SMACliteWrapper(**kwargs)


def gymma_fn(**kwargs) -> MultiAgentEnv:
    assert "common_reward" in kwargs and "reward_scalarisation" in kwargs
    return GymmaWrapper(**kwargs)


REGISTRY = {}

# REGISTRY["smaclite"] = smaclite_fn
REGISTRY["gymma"] = gymma_fn


# Register SMAC dynamically if required
def register_smac():
    from .smac_wrapper import SMACWrapper

    def smac_fn(**kwargs) -> MultiAgentEnv:
        kwargs = __check_and_prepare_smac_kwargs(kwargs)
        return SMACWrapper(**kwargs)

    REGISTRY["sc2"] = smac_fn


# Register SMACv2 dynamically if required
def register_smacv2():
    from .smacv2_wrapper import SMACv2Wrapper

    def smacv2_fn(**kwargs) -> MultiAgentEnv:
        kwargs = __check_and_prepare_smac_kwargs(kwargs)
        return SMACv2Wrapper(**kwargs)

    REGISTRY["sc2v2"] = smacv2_fn
    