import torch
import torch.nn as nn
import torch.nn.functional as F
  
class BollingerClassifier(nn.Module):
    def __init__(self, num_classes=3): # 
        super(BollingerClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1)
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
