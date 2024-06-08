from gym.envs.registration import register
from copy import deepcopy

from . import datasets


register(
    id='btc-futures-01',
    entry_point='gym_futures_trading.envs:TradingEnv',
    kwargs={
        'df': deepcopy(datasets.FUTURES_01),
        'window_size': 24,
        'frame_bound': (24, len(datasets.FUTURES_01))
    }
)

register(
    id='btc-futures-02',
    entry_point='gym_futures_trading.envs:TradingEnv',
    kwargs={
        'df': deepcopy(datasets.FUTURES_02),
        'window_size': 24,
        'frame_bound': (24, len(datasets.FUTURES_02))
    }
)

register(
    id='btc-futures-03',
    entry_point='gym_futures_trading.envs:TradingEnv',
    kwargs={
        'df': deepcopy(datasets.FUTURES_03),
        'window_size': 24,
        'frame_bound': (24, len(datasets.FUTURES_03))
    }
)

register(
    id='btc-futures-04',
    entry_point='gym_futures_trading.envs:TradingEnv',
    kwargs={
        'df': deepcopy(datasets.FUTURES_04),
        'window_size': 24,
        'frame_bound': (24, len(datasets.FUTURES_04))
    }
)
