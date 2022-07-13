import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
import madgrad
import torch
from sklearn.metrics import balanced_accuracy_score
import numpy as np

import sys  
sys.path.insert(0, './code')
from RADAM import RAdam

import logging
import warnings

logging.getLogger("lightning").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


class PLModel(pl.LightningModule):
    def __init__(self, name, model,loss = None, lastLayer=False, num_epochs = 0, path = None, lr = 0.001, wd = 0.0, reduced = None, optimName = 'noOptimName'):
        super(PLModel, self).__init__()

        self.optimName = optimName
        self.maxValAcc = 0
        self.lr = lr
        
        if loss is None:
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = loss
            
        if reduced is None:
            self.reduced = ''
        else:
            self.reduced = reduced
            
        if path is None:
            self.PATH = 'skin/supervised/'+self.reduced+self.optimName+name+'.ckpt'
        
        self.wd = wd
        self.name = name
        self.epoch = num_epochs
        self.model = model
        
        self.isLL = False
        
        if lastLayer:
            if name == 'EfficientNet':
                numFeat = self.model._fc.in_features
                self.model._fc = nn.Linear( numFeat, numFeat * 2)
            else:
                numFeat = self.model.fc.in_features
                self.model.fc = nn.Linear( numFeat, numFeat * 2)
            self.fc = nn.Linear( numFeat * 2, numFeat)
            self.lfc = nn.Linear( numFeat, 8)
            self.isLL = True

        self.writer = SummaryWriter("runs/"+self.reduced+self.optimName+ str(self.isLL)+ self.name)
        
        
        
    def forward(self, x):
        x = self.model(x)
        
        if self.isLL:
            x = self.fc(x)
            x = self.lfc(x)
            
        return x
    
    def configure_optimizers(self):
        if self.optimName == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), lr=0.01)
        if self.optimName == 'MADGRAD':
            optimizer = madgrad.MADGRAD(self.parameters(), lr = self.lr, momentum = 0.9, weight_decay = self.wd, eps = 1e-06)
        else:
            optimizer = RAdam(self.parameters(), lr=self.lr , weight_decay= self.wd)
            
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, min_lr=1e-7, patience=10)
        print(optimizer)
        return {
           'optimizer': optimizer,
           'scheduler': scheduler,
           'monitor': 'val_loss'
       }
    
    def training_step(self, batch, batch_idx):
        acc = 0
        x, y = batch
        
        #Siamo all'interno della classe quindi per rifermi alla rete utilizzo self (al posto di CNN(x))
        output = self(x)
        
        J = self.loss(output, y)
        
        #print(y)
        
        for i in range(x.shape[0]):
            #print(y[i], output[i])
            if y[i] == torch.argmax(output[i]):
                acc += 1
        
        pbar = {'train_acc' : acc/x.shape[0]}
        
        return {'loss' : J,
                'progress_bar': pbar}
    
    def training_epoch_end(self, train_step_outputs):
        avg_train_loss = torch.tensor([x['loss'] for x in train_step_outputs]).mean()
        avg_train_acc = torch.tensor([x['progress_bar']['train_acc'] for x in train_step_outputs]).mean()
        
        self.writer.add_scalar('training loss', avg_train_loss, self.epoch)
        self.writer.add_scalar('training accuracy', avg_train_acc, self.epoch)
        
        self.epoch += 1
        
        pbar = {'train_acc' : avg_train_acc }
        
        return {'train_loss' : avg_train_loss,
                'progress_bar': pbar}
    
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        acc = 0
        x, y = batch

        output = self(x)
        
        J = self.loss(output, y)
        
        for i in range(x.shape[0]):
            if y[i] == torch.argmax(output[i]):
                acc += 1
        
        _, predicted = torch.max(output, 1)

        pbar = {'val_acc' : acc/x.shape[0],'bal_val_acc' : balanced_accuracy_score(np.array(y.to('cpu')), np.array(predicted.to('cpu')))}
        
        return {'loss' : J,
                'progress_bar': pbar}

    @torch.no_grad()
    def validation_epoch_end(self, val_step_outputs):
        avg_val_loss = torch.tensor([x['loss'] for x in val_step_outputs]).mean()
        avg_val_acc = torch.tensor([x['progress_bar']['val_acc'] for x in val_step_outputs]).mean()
        bal_avg_val_acc = torch.tensor([x['progress_bar']['bal_val_acc'] for x in val_step_outputs]).mean()
        
        self.writer.add_scalar('validation loss', avg_val_loss, self.epoch)
        self.writer.add_scalar('validation accuracy', avg_val_acc, self.epoch)
        self.writer.add_scalar('balanced validation accuracy', bal_avg_val_acc, self.epoch)
        
        pbar = {'val_acc' : avg_val_acc,'bal_val_acc' : bal_avg_val_acc}
        
        
        if bal_avg_val_acc > self.maxValAcc:
            self.maxValAcc = bal_avg_val_acc
            torch.save(self.model.state_dict(), self.PATH)
        
        return {'val_loss' : avg_val_loss,
                'progress_bar': pbar}

    def closeWriter(self):
        self.writer.close()
