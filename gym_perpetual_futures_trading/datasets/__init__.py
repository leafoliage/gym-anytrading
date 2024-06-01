from .utils import load_dataset as _load_dataset


# Load FOREX datasets
BTC_PERPETUAL = _load_dataset('FOREX_EURUSD_1H_ASK', 'Time')
