import pandas as pd
import numpy as np
import yfinance as yf
import gc
from datetime import datetime
import os
import pickle

from sklearn.preprocessing import StandardScaler
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

from pyts.image import GramianAngularField

class PrepareDataset:

    def __init__(self, config):
        self.cfg = config
        self.data_dir = self.cfg.data_dir
        self.tickers = self.cfg.tickers
        self.extension = ''.join([i[0].lower() for i in self.cfg.cols])
        if self.cfg.label_on_price_change:
            self.extension = 'pc_' + self.extension
        print('Preparing data to train the DQN model...')

    def fetch_from_yahoo_finance(self, ticker, start, end):
        df = yf.download(tickers=ticker,
                         start=start,
                         end=None,
                         progress=False,
                         interval='1d')
        return df.fillna(method='bfill')
    
    
    def normalize(self, df, cols):
        df.reset_index(inplace = True)
        # Normalize with max value
        df['Date'] = pd.to_datetime(df['Date'])
        df['year'] = df['Date'].dt.year
        df['quarter'] = df['Date'].dt.quarter
        df['original_close'] = df['Close']
        df['norm_close'] = df['Close']
        df['pct_change'] = df['Close'].pct_change()

        dfs = []
        for y in df['year'].unique():
            for q in df['quarter'].unique():
                mdf = df[(df['year'] == y) & (df['quarter'] == q)]
                for c in cols + ['norm_close']:
                    min_val = min(mdf[c])
                    max_val = max(mdf[c])
                    mdf[c] = (mdf[c] - min_val) / (max_val - min_val)
                dfs.append(mdf)

        # normalized
        normalized_df = pd.concat(dfs)
        normalized_df.sort_values(by = 'Date', ascending = True, inplace = True)

        return normalized_df

    def make_dataset(self, df, ticker, kind):
        ohlc_columns = self.cfg.cols
        df = self.normalize(df, cols = ohlc_columns)             # normalize by max value
        w = self.cfg.window                              # window size
        data = []
        gadf = GramianAngularField(image_size=w, method='difference', sample_range=(0, 1))
        for i in range(len(df)-w-1, -1, -1):
            gadf_images = [gadf.transform([df[c][i:i+w]]) for c in ohlc_columns]
            patches = np.vstack([df[c][i:i+w] for c in ohlc_columns])                   # stack raw_data
            gadf_3d = np.vstack(gadf_images)
            
            typical_prices = (df['High'][i:i+w] + df['Low'][i:i+w] + df['Close'][i:i+w]) / 3
            if self.cfg.label_on_price_change:
                label = self.make_label_price_change(close_price=df['Close'][i+w-1],
                                                     next_close_price = df['Close'][i+w])
            else:
                label = self.make_label(typical_prices=typical_prices, next_close_price=df['Close'][i+w])
            combined_gadf_image = np.sum(gadf_images, axis=0)/len(gadf_images)
            norm_close = df['norm_close'][i+w-1]
            original_close = df['original_close'][i+w-1]
            
            data.append((i, combined_gadf_image, gadf_3d, patches, label, norm_close, original_close))  # single element in a list

        # save
        with open(f'{self.data_dir}/{kind}_{ticker}_{self.extension}.pkl', 'wb') as f:
            pickle.dump(data, f)

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

    def make_label_price_change(self, close_price, next_close_price):
        pct_change = (next_close_price - close_price) / close_price
        if pct_change >= 5/100:
            label = 1
        elif pct_change <= -5/100:
            label = 2
        else:
            label = 0
        return label

    def load(self):
        self.train, self.valid, self.test = {}, {}, {}          #key (ticker), values (list of images, labels,...)
        if os.path.exists(f'{self.data_dir}/train_{self.tickers[0]}_{self.extension}.pkl'):
            for ticker in self.tickers:
                with open(f'{self.data_dir}/train_{ticker}_{self.extension}.pkl', 'rb') as f:
                    self.train[ticker] = pickle.load(f)
                with open(f'{self.data_dir}/valid_{ticker}_{self.extension}.pkl', 'rb') as f:
                    self.valid[ticker] = pickle.load(f)
                with open(f'{self.data_dir}/test_{ticker}_{self.extension}.pkl', 'rb') as f:
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
        self.is_2dcnn = self.cfg.train_2dcnn
        self.data = data
        self.load_data()

    def load_data(self):
        self.images = []
        self.labels = []
        for _, ticker_data in self.data.items():
            for (idx, gadf_image, gadf_3d, patch, label, _, _) in ticker_data:
                sample = (gadf_image if self.cfg.model_arch == '2d' else gadf_3d) if self.cfg.train_gadf_image else patch
                if self.cfg.centered_zero:
                    sample = (sample - 0.5) * 2
                self.images.append(sample if self.is_2dcnn else sample.flatten())   # in 1dcnn convert the image to array of raw pixels
                self.labels.append(label)
        
        # Training with sample of data (use it while running on CPU)
        if self.cfg.sample_run:
            self.images = self.images[:50]
            self.labels = self.labels[:50]
            
        # from collections import Counter
        # print(Counter(self.labels))

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
