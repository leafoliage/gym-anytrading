from gym.envs.registration import register
from copy import deepcopy

from . import datasets


register(
    id='btcperp-v0',
    entry_point='gym_perpetual_futures_trading.envs:BtcPerpEnv',
    kwargs={
        'df': deepcopy(datasets.BTC_PERPETUAL),
        'window_size': 24,
        'frame_bound': (24, len(datasets.BTC_PERPETUAL))
    }
)
