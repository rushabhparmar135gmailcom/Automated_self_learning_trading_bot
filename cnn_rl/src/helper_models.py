import torch
import torch.nn as nn
import torch.nn.functional as F

class BollingerClassifier(nn.Module):
    def __init__(self, config, num_classes=3):
        super(BollingerClassifier, self).__init__()
        self.config = config
        in_channels = len(self.config.cols) if self.config.model_arch == '3d' else (1 if self.config.train_gadf_image else 4)
        
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu3 = nn.ReLU()
        #self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.fc1 = nn.Linear(16 * 6 * 6, 128)
        self.relu4 = nn.ReLU()
        self.dropout1 = nn.Dropout(0)
        
        self.fc2 = nn.Linear(128, 32)
        self.relu5 = nn.ReLU()
        self.dropout2 = nn.Dropout(0)
        
        self.fc3 = nn.Linear(32, num_classes)

    def loss_fn(self, outputs, y):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, y)
        return loss
    
    def forward(self, x, y):
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = self.relu3(self.conv3(x))
        x = x.view(-1, 16 * 6 * 6)
        x = self.relu4(self.fc1(x))
        x = self.relu5(self.fc2(x))
        logits = self.fc3(x)
        loss = self.loss_fn(logits, y)
        return x, logits, loss
    
#### 1D CNN
class Baseline1DCNN(nn.Module):
    
    '''
        Baseline version of the model architecture is a single layer 
        conv1d > max pool > fully connected (2)
    '''
    
    def __init__(self, config, num_classes):
        super(Baseline1DCNN, self).__init__()
        self.config = config
        num_features = (config.window * config.window) if config.train_gadf_image else config.window
        in_channels = 1 if config.train_gadf_image else len(self.config.cols)
        
        self.conv1 = nn.Conv1d(in_channels= in_channels, out_channels=16, kernel_size = 3, stride = 1, padding = 1)
        self.maxpool1 = nn.AvgPool1d(kernel_size=2, stride = 2)
        
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=2, kernel_size = 3, stride = 1, padding = 1)
        
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(num_features, num_classes)
        
    def loss_fn(self, outputs, y):
        criterion = nn.CrossEntropyLoss()
        loss = criterion(outputs, y)
        return loss
        
    def forward(self, x, y):
        # input x is of shape [16, 576] where 16 is the batchsize 
        # it must be reshaped to work with 1DCNN -> [16, 1, 576]
        x = x.unsqueeze(1) if self.config.train_gadf_image else x.reshape(x.size(0), len(self.config.cols), 24)
        x = self.maxpool1(self.relu(self.conv1(x)))
        x = self.relu(self.conv2(x))
        
        x = self.flatten(x)
        
        logits = self.fc1(x)
        loss = self.loss_fn(logits, y)
        return x, logits, loss
    
#### POLICY NETWORK
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PolicyNetwork, self).__init__()
        
        '''
            input_size  : last fully connnected layer x number of assets
            output_size : weights of the assets
        '''
        
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 4)
        self.fc3 = nn.Linear(hidden_size // 4, output_size)
        
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        
    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        logits = self.fc3(x)
        weights = torch.sigmoid(logits)
        return weights

    
if __name__ == "__main__":
    model = BollingerClassifier(num_classes=3)
    print(model)

