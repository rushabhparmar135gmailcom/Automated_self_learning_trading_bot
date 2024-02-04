import numpy as np

### ENV
class PortfolioEnvironment:
    def __init__(self, config, norm_price_data, price_data, images, labels,window_size = 1000):
        
        self.config = config
        self.norm = norm_price_data.values
        self.data = price_data.values
        self.images = images
        self.labels = labels
        self.num_assets = len(self.config.tickers)
        self.total_steps = price_data.shape[0]              # unitl the end of time period
        self.target_assets = len(self.config.tickers)
        self.initial_balance = 1e6                          # million
        self.riskfree_rate = 0.02                           # riskfree rate: 2%
        self.trading_cost = 0.25/100.                       # Trading cost : 0.25%
        self.window_size = window_size
        self.return_history = []
        
        # reset environment
        self.reset()
        print(f'Initial portfolio balance: {self.balance}')
        
    def reset(self):
        self.balance = self.initial_balance
        self.weights = np.zeros(self.target_assets)  # weights are zeroed
        self.current_step = 0
        self.return_history = []
        return self.get_observation()
    
    def get_observation(self):
        # get current state
        observation = (self.data[self.current_step], 
                       self.norm[self.current_step],
                       self.images[self.current_step], 
                       self.labels[self.current_step])
        return observation
    
    def step(self, weights, trade_signals):
        
        if self.current_step >= self.total_steps:
            raise Exception('End of data.')
        
        # take an action
        # objective is maximize portfolio returns
        
        ### STEP 1: Estimate returns
        current_price = self.data[self.current_step][:self.target_assets]
        prev_price = self.data[self.current_step - 1][:self.target_assets]
        
        # Update portfolio balance on asset weights
        #portfolio_value = self.balance + np.dot(prev_price, self.weights)                       # previous step portfolio value
        
        # for HOLD signal use the previous weights, while ensuring a position in the asset
        if sum(self.weights) > 0:
            delta_w = [0 if s == 2 else max(0, w - prev_w) for s, prev_w, w in zip(trade_signals, self.weights, weights) if prev_w >= 0] # when the signal is hold, we dont change the weights
            txn_w = [0 if s == 2 else dw for s, dw in zip(trade_signals, delta_w)]
            weights = [prev_w if s == 2 else w for s, prev_w, w in zip(trade_signals, self.weights, weights) if prev_w >= 0]
            weights, delta_w = np.array(weights), np.array(delta_w)
        else:
            txn_w, delta_w = weights, weights
            
        txn_cost = np.sum(np.dot(txn_w, current_price) * self.trading_cost)
        new_portfolio_value = self.balance + np.dot(current_price, delta_w) - txn_cost
        
        # Update portfolio weights
        self.weights = weights
        
        # Calculate reward as portfolio returns
        returns = (new_portfolio_value - self.initial_balance) / self.initial_balance
        
        ### Step 2: Estimate risk adjusted return
        rf = self.riskfree_rate / 252           # Daily data
        reward = returns - rf
        
        # Update balance with new portfolio value
        self.balance = new_portfolio_value
        
        # Step 3: Take next step
        next_step = self.current_step + 1
        
        # Step 4: Check if the episode is done
        done = (next_step >= self.total_steps - 1)
        self.current_step = next_step
        
        return self.balance, returns, reward, done