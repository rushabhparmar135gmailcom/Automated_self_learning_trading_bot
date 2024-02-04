import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp              # mixed precision
from torch import autocast
from torchsummary import summary
from datetime import datetime

from sklearn.metrics import f1_score

## Local imports
from helper_models import BollingerClassifier
from helper_dataset import dataloaders

import warnings
warnings.filterwarnings('ignore')

#### Evaluate
def evaluate(model, dataloader, device):
    student_ids = []
    y_trues = []
    y_preds = []
    batch_loss_list = []
    model.eval()
    with torch.no_grad():
        for inputs, targets in dataloader:
            _, logits, loss = model(inputs.to(device), targets.to(device))
            probs = F.softmax(logits, dim=1)
            _, labels = torch.max(probs, dim=1)
            batch_loss_list.append(loss.item())
            
            # save
            y_trues.extend(targets.to('cpu').numpy().tolist())
            y_preds.extend(labels.to('cpu').numpy().tolist())
    
    y_trues = np.array(y_trues)
    y_preds = np.array(y_preds)
    
    loss = np.mean(batch_loss_list)
    score = f1_score(y_trues, y_preds, average = 'weighted')
    return score, loss

#### Trainer
def trainer(config, model, train_loader, valid_loader, optimizer, scheduler):
    
    def update_que():
        que.set_postfix({
            'batch_loss'        : f'{loss.item():4f}',
            'epoch_loss'        : f'{np.mean(batch_loss_list):4f}',
            'learning_rate'     : optimizer.param_groups[0]["lr"],
            })
    
    def save_checkpoint(model, epoch, best = False):
        if best:
            save_path = os.path.join(config.dest_path, f'model.pth')
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
                }
            torch.save(checkpoint, save_path)
            print(f'>>> [{datetime.now()}] - Checkpoint and predictions saved')
        
    def dis(x): return f'{x:.6f}'
        
    def run_evaluation_sequence(ref_score, counter):
        
        def print_result():
            print('')
            text =  f'>>> [{datetime.now()} | {epoch + 1}/{NUM_EPOCHS} | Early stopping counter {counter}] \n'
            text += f'    loss          - train: {dis(train_loss)} valid: {dis(valid_loss)} \n'
            text += f'    f1-score      - train: {dis(train_f1)} valid: {dis(valid_f1)} \n'
            text += f'    learning rate        : {optimizer.param_groups[0]["lr"]:.5e}'
            print(text + '\n')
        
        # Evaluation
        train_f1, train_loss = evaluate(model, train_loader, device) 
        valid_f1, valid_loss = evaluate(model, valid_loader, device)
        
        # append results
        lr =  optimizer.param_groups[0]["lr"]
        results.append((epoch, train_loss, valid_loss, train_f1, valid_f1, lr))
        
        # Learning rate scheduler
        scheduler.step(valid_f1)
        
        ### Save checkpoint
        if ((epoch + 1) > config.save_epoch_wait) and (config.save_checkpoint):
            save_checkpoint(model, epoch, best = valid_f1 > ref_score)
        
        # Tracking early stop
        counter = 0 if valid_f1 > ref_score else counter + 1
        ref_score = max(ref_score, valid_f1)
        done = counter >= config.early_stop_count
        
        # show results
        print_result()
        
        return ref_score, counter, done 
    
    ### MIXED PRECISION
    scaler = amp.GradScaler()
    
    results = []
    device = config.device
    NUM_EPOCHS = config.num_epochs
    iters_to_accumlate = config.iters_to_accumlate
    
    # dummy value for placeholders
    ref_score, counter = 1e-3, 0
    train_loss, valid_loss, train_f1, valid_f1 = 0, 0, 0, 0
    for epoch in range(NUM_EPOCHS):
        model.train()
        batch_loss_list = []

        que = tqdm(enumerate(train_loader), total = len(train_loader))
        for i, (images, targets) in que:
            
            ###### TRAINING SECQUENCE            
            with autocast(device_type = str(device), dtype = torch.float16):
                _, _, loss = model(images.to(device), targets.to(device))            # Forward pass
                loss = loss / iters_to_accumlate
            
            # - Accmulates scaled gradients    
            scaler.scale(loss).backward()           # scale loss
            
            if (i + 1) % iters_to_accumlate == 0:
                scaler.step(optimizer)                  # step
                scaler.update()
                optimizer.zero_grad()
            #######
            
            batch_loss_list.append(loss.item())
            
            # Update que status
            update_que()
        
        ### Run evaluation sequence
        ref_score, counter, done = run_evaluation_sequence(ref_score, counter)
        if done:
            return results
            
    return results

def save_config(config, path):
    with open(os.path.join(path, 'config.pkl'), 'wb') as f:
        pickle.dump(config, f)

def train(config):
    device = config.device
    
    config.dest_path = os.path.join(config.models_dir, config.model_name)
    os.makedirs(config.dest_path, exist_ok=True)
    
    # define model
    model = BollingerClassifier(num_classes=3)
    model.to(device)
    
    # optmizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr = config.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'max', factor=0.5, patience=4)
    
    # dataloaders
    train_loader, valid_loader, _ = dataloaders(config)
    
    # Trainer
    results = trainer(config, model, train_loader, valid_loader, optimizer, scheduler)
    
    ### SAVE RESULTS
    with open(os.path.join(config.dest_path, 'results.pkl'), 'wb') as f:
        pickle.dump(results, f)


if __name__ == '__main__':
    from helper_config import Config
    
    config = Config()
    config.data_dir = 'data'      
    config.models_dir = 'models/cnn_predictor' 
    config.model_name = 'baseline'
    config.train_batch_size = 16
    config.iters_to_accumlate = 1
    config.sample_run = False
    config.learning_rate = 1e-4
    config.num_epochs = 100
    config.save_epoch_wait = 1    
    config.early_stop_count = 10
    config.save_checkpoint = True
    
    results = train(config)
