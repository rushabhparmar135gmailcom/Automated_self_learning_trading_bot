from helper_config import Config
from helper_env import PortfolioEnvironment
from helper_agent import DQLAgent

import datetime
import pandas as pd
cfg = Config()
cfg.data_dir = 'data'
cfg.models_dir = 'models/rl'
cfg.base_model_path = 'models/cnn_predictor/baseline/model.pth'
target_assets = len(cfg.tickers)

### RL / POLICY NETWORK CONFIGURATION
cfg.input_dim       = len(cfg.tickers) * (4)            # 3 logits from CNN network and 1 norm close price
cfg.output_dim      = len(cfg.tickers)
cfg.learning_rate   = 1e-3
cfg.gamma           = 0.99
cfg.epsilon         = 0.5
cfg.epsilon_decay   = 0.995
cfg.epsilon_min     = 0.01
cfg.reply_capacity  = 10000
cfg.batch_size      = 32
cfg.hidden_size     = 512
cfg.riskfree_rate   = 0.02

cfg.initial_balance = 1000000    

prep = RLDataset(cfg)
df_norm_close, df_prices, images, labels = prep.prepare(kind = 'train')

env = PortfolioEnvironment(cfg, df_norm_close.head(50), df_prices.head(50), images, labels, window_size = 2000)
agent = DQLAgent(cfg, name = 'baseline')
results = agent.train(env, episodes = 10)