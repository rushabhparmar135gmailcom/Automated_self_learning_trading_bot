import pandas as pd
import numpy as np
import yfinance as yf
import gc
from datetime import datetime
import os
import pickle

from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader

from pyts.image import GramianAngularField

class PrepareDataset: # We have this class, which prepares the dataset for initial training, which is images, and labels and normalized closed price

    def __init__(self, config):
        self.cfg = config # We create config class object here
        self.data_dir = self.cfg.data_dir  # update the data_directory attribute, with the path name defined in the class
        self.tickers = self.cfg.tickers # list of tickers, go here
        print('Preparing data to train the DQN model...')

    def fetch_from_yahoo_finance(self, ticker, start, end): # using this function, initially we fetch data from yahoo finance
        # using the yf library
        df = yf.download(tickers=ticker,
                         start=start,
                         end=None,
                         progress=False,
                         interval='1d')
        return df.fillna(method='bfill')
    
    # we normalize the data using Min Max normalization, based on each quarter
    def normalize(self, df):
        df.reset_index(inplace = True)
        ohlc_columns = ['Open', 'High', 'Low', 'Close']
        # Normalize with max value
        df['Date'] = pd.to_datetime(df['Date'])
        df['year'] = df['Date'].dt.year
        df['quarter'] = df['Date'].dt.quarter
        df['original_close'] = df['Close']
        df['norm_close'] = df['Close']

        dfs = []
        for y in df['year'].unique():
            for q in df['quarter'].unique():
                mdf = df[(df['year'] == y) & (df['quarter'] == q)]
                for c in ohlc_columns + ['norm_close']:
                    min_val = min(mdf[c])
                    max_val = max(mdf[c])
                    mdf[c] = (mdf[c] - min_val) / (max_val - min_val)
                dfs.append(mdf)

        # normalized
        normalized_df = pd.concat(dfs)
        normalized_df.sort_values(by = 'Date', ascending = True, inplace = True)

        return normalized_df
# now that we have normalized dataset, we create GASF images, out of the dataset, 
# we save images, label, close price and normalized close price, and kind, which is train, validate or test as a list in pickle file
    def make_dataset(self, df, ticker, kind):
        ohlc_columns = ['Open', 'High', 'Low', 'Close']
        df = self.normalize(df)             # normalize by max value
        w = 24                              # window size
        data = []
        gasf = GramianAngularField(image_size=w, method='difference', sample_range=(0, 1))
        for i in range(len(df)-w-1, -1, -1):
            gasf_images = [gasf.transform([df[c][i:i+w]]) for c in ohlc_columns]
            typical_prices = (df['High'][i:i+w] + df['Low'][i:i+w] + df['Close'][i:i+w]) / 3
            label = self.make_label(typical_prices=typical_prices, next_close_price=df['Close'][i+w])
            combined_gasf_image = np.sum(gasf_images, axis=0)/len(gasf_images)
            # combined_gasf_image = np.vstack(gasf_images)
            norm_close = df['norm_close'][i+w-1]
            original_close = df['original_close'][i+w-1]
            data.append((i, combined_gasf_image, label, norm_close, original_close))

        # save
        with open(f'{self.data_dir}/{kind}_{ticker}.pkl', 'wb') as f:
            pickle.dump(data, f)
# this function, creates a bollinger band using mean +/- std_deviation*1.5, and based on it labels the images
# if the next close price is above the bollinger band, we sell the stock, below, we buy the stock or hold
    def make_label(self, typical_prices, next_close_price, n=1.5):
        # https://www.investopedia.com/terms/b/bollingerbands.asp
        mean_price = np.mean(typical_prices)
        std_dev = np.std(typical_prices)
        if next_close_price >= mean_price + std_dev * n:
            label = 0               # price appricated      (Sell signal)
        elif next_close_price <= mean_price - std_dev * n:
            label = 1               # price depreciated     (Buy opportunity)
        else:
            # price indifferent     (indifferent to make decision)
            label = 2
        return label

    def load(self): # here we load the data saved in pickle files as train_ticker.pkl and so on, in their respctive dictionaries
        # where the keys are the ticker names and dictionary name is as train, valid and test
        self.train, self.valid, self.test = {}, {}, {}
        if os.path.exists(f'{self.data_dir}/train_{self.tickers[0]}.pkl'):
            for ticker in self.tickers:
                with open(f'{self.data_dir}/train_{ticker}.pkl', 'rb') as f:
                    self.train[ticker] = pickle.load(f)
                with open(f'{self.data_dir}/valid_{ticker}.pkl', 'rb') as f:
                    self.valid[ticker] = pickle.load(f)
                with open(f'{self.data_dir}/test_{ticker}.pkl', 'rb') as f:
                    self.test[ticker] = pickle.load(f)
        else:
            # load data from Yahoo! Finance
            for i, ticker in enumerate(self.tickers):
                print(f'[{datetime.now()}] [{i}/{len(self.tickers)}] Processing for ticker: {ticker}')
                df = self.fetch_from_yahoo_finance(ticker=ticker, start='2016-01-01', end='2021-12-31')
                for kind, dt_range in self.cfg.splits.items():
                    start, end = dt_range['start'], dt_range['end']
                    self.make_dataset(df=df.loc[start:end, :],
                                      ticker=ticker,
                                      kind=kind)
            # make dataset
            self.load()

# DATALOADER FOR THE MODEL
class MyDataset(Dataset):
    def __init__(self, config, data):
        self.cfg = config
        self.data = data
        self.load_data()

    def load_data(self):
        self.images = []
        self.labels = []
        for _, ticker_data in self.data.items():
            for (idx, gasf_image, label, _, _) in ticker_data:
                self.images.append(gasf_image)
                self.labels.append(label)
        if self.cfg.sample_run:
            self.images = self.images[:50]
            self.labels = self.labels[:50]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.tensor(self.images[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return image, label

# DATALOADERS

def dataloaders(cfg):

    prep = PrepareDataset(config=cfg)
    prep.load()

    train = MyDataset(cfg, prep.train)
    valid = MyDataset(cfg, prep.valid)
    test = MyDataset(cfg, prep.test)

    train_loader = DataLoader(train,
                              batch_size=cfg.train_batch_size,
                              shuffle=True,
                              drop_last=False)

    valid_loader = DataLoader(valid,
                              batch_size=cfg.valid_batch_size,
                              shuffle=False)

    test_loader = DataLoader(test,
                             batch_size=cfg.test_batch_size,
                             shuffle=False)

    return train_loader, valid_loader, test_loader


# DATASET FOR RL
class RLDataset:
    def __init__(self, config):
        self.config = config

    def prepare(self, kind='train'):
        # load dataset
        data = {}
        for tick in self.config.tickers:
            with open(f'{self.config.data_dir}/{kind}_{tick}.pkl', 'rb') as f:
                data[tick] = pickle.load(f)[::-1]       # reverse

        # make images, labels, and close_price dataset
        # Close price is dataframe
        norm_close = {}
        original_close = {}
        for tick in self.config.tickers:
            norm_close[tick]        = [item[3] for item in data[tick]]
            original_close[tick]    = [item[4] for item in data[tick]]
        df_norm_close = pd.DataFrame.from_dict(norm_close)
        df_org_close = pd.DataFrame.from_dict(original_close)

        # Images and labels are list of arrays
        images, labels = [], []
        for idx in range(len(df_norm_close)):
            batch_images = np.array(
                [data[tick][idx][1] for tick in self.config.tickers if data[tick][idx][0] == idx])
            batch_labels = np.array(
                [data[tick][idx][2] for tick in self.config.tickers if data[tick][idx][0] == idx])
            images.append(batch_images)
            labels.append(batch_labels)

        return df_norm_close, df_org_close, images, labels
