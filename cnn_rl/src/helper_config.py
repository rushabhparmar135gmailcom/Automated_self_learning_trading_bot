import torch

class Config:

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # REPO
    data_dir = '../data'
    model_dir = '..models/baseline'

    # MICROS
    tickers = ['RELIANCE.NS', 'TCS.NS', 'INFY.NS',
               'HINDUNILVR.NS', 'ICICIBANK.NS', 'HDFCBANK.NS',
               'BHARTIARTL.NS', 'KOTAKBANK.NS', 'WIPRO.NS',
               'LT.NS'] # 'HDFC.NS',

    # Splits
    splits = {
        'train': {
            'start': '2016-01-02',
            'end': '2019-12-31'
        },
        'valid': {
            'start': '2020-01-02',
            'end': '2020-12-31'
        },
        'test': {
            'start': '2021-01-02',
            'end': '2021-12-31'
        }
    }
    
    cols = ['Open', 'High', 'Low', 'Close']
    
    # Data prep
    window = 24

    train_batch_size = 16
    valid_batch_size = 32
    test_batch_size = 32

    num_epochs = 100
    iters_to_accumlate = 1
    
    centered_zero = False
    
    # Model params
    train_2dcnn = True
    train_gadf_image = True
    label_on_price_change = False
