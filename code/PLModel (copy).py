import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
import madgrad
import torch
import numpy as np

import sys  
sys.path.insert(0, './code')
from RADAM import RAdam

import logging
import warnings

logging.getLogger("lightning").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


class PLModel(pl.LightningModule):
    def __init__(self, name, model,loss = None, lastLayers=None, num_epochs = 0, path = None, lr = 0.001, wd = 0.0, reduced = None, optimName = 'noOptimName', BYOL = False):
        super(PLModel, self).__init__()

        self.maxTestAcc = 0
        self.optimName = optimName
        self.maxValAcc = 0
        self.lr = lr
        self.wd = wd
        self.name = name
        self.epoch = num_epochs
        self.model = model
        self.isLL = None
        self.labelName = ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"]

        if BYOL:
            self.type = 'BYOL'
        else:
            self.type = 'STANDARD'
        
        if loss is None:
            print('No loss specified, using default')
            self.loss = nn.CrossEntropyLoss()
        else:
            self.loss = loss
            
        if reduced is None:
            self.reduced = ''
        else:
            self.reduced = reduced
            
        if path is None:
            self.PATH = 'skin/supervised/'+self.type+self.reduced+self.optimName+self.name+'.ckpt'
        
        if lastLayers is not None:
            if name == 'EfficientNet':
                numFeat = self.model._fc.in_features
                self.model._fc = nn.Linear( numFeat, numFeat)
            else:
                numFeat = self.model.fc.in_features
                self.model.fc = nn.Linear( numFeat, numFeat)
            self.isLL = lastLayers

        self.writer = SummaryWriter("runs/"+self.type+self.reduced+self.optimName+ self.name)
        
        
        
    def forward(self, x):
        x = self.model(x)
        
        if self.isLL is not None:
            x = self.isLL(x)
            
        return x
    
    def configure_optimizers(self):
        if self.optimName == 'SGD':
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr)
        elif self.optimName == 'MADGRAD':
            optimizer = madgrad.MADGRAD(self.parameters(), lr = self.lr, momentum = 0.9, weight_decay = self.wd, eps = 1e-06)
        else:
            optimizer = RAdam(self.parameters(), lr=self.lr , weight_decay= self.wd)
            
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, min_lr=1e-7, patience=10)
        print(optimizer)
        return {
           'optimizer': optimizer,
           'scheduler': scheduler,
           'monitor': 'train_loss'
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
        n_class_correct = torch.zeros(len(self.labelName)).to('cuda')
        n_class_samples = torch.zeros(len(self.labelName)).to('cuda')

        output = self(x)
        
        J = self.loss(output, y)
        
        for i in range(x.shape[0]):
            label = torch.argmax(output[i])
            if y[i] == label:
                acc += 1
                n_class_correct[label] += 1
            n_class_samples[y[i]] += 1

        pbar = {'val_acc' : acc/x.shape[0]}

        for i in range(len(self.labelName)):
           pbar[self.labelName[i]] = n_class_correct[i]
           pbar[self.labelName[i] + 'TOT'] = n_class_samples[i]

        return {'loss' : J,
                'progress_bar': pbar}

    @torch.no_grad()
    def validation_epoch_end(self, val_step_outputs):
        n_class_correct = torch.zeros(len(self.labelName)).to('cuda')
        n_class_samples = torch.zeros(len(self.labelName)).to('cuda')
        bal_acc = 0.

        avg_val_loss = torch.tensor([x['loss'] for x in val_step_outputs]).mean()
        avg_val_acc = torch.tensor([x['progress_bar']['val_acc'] for x in val_step_outputs]).mean()

        for i in range(len(self.labelName)):
           n_class_correct[i] = torch.tensor([x['progress_bar'][self.labelName[i]] for x in val_step_outputs]).sum()
           n_class_samples[i] = torch.tensor([x['progress_bar'][self.labelName[i] + 'TOT'] for x in val_step_outputs]).sum() 
        
        self.writer.add_scalar('validation loss', avg_val_loss, self.epoch)
        self.writer.add_scalar('validation accuracy', avg_val_acc, self.epoch)
        self.writer.add_scalar('learning rate', self.getOptimizer().optimizer.param_groups[0]['lr'], self.epoch)

        for i in range(len(self.labelName)):
            self.writer.add_scalar('validation accuracy '+self.labelName[i], n_class_correct[i]/n_class_samples[i], self.epoch)
            if n_class_samples[i] != 0:
                bal_acc += n_class_correct[i]/n_class_samples[i]

        bal_acc = bal_acc/len(self.labelName)
        self.writer.add_scalar('balanced validation accuracy', bal_acc, self.epoch)
        pbar = {'val_acc' : avg_val_acc, 'bal_acc' : bal_acc}
        
        
        if bal_acc > self.maxValAcc:
            self.maxValAcc = avg_val_acc
            torch.save(self.model.state_dict(), self.PATH)
        
        return {'val_loss' : avg_val_loss,
                'progress_bar': pbar}

    def closeWriter(self):
        self.writer.close()

    def trainLastLayers(self):
        for param in self.model.parameters():
            param.requires_grad = False
        if self.name == 'EfficientNet':
            for param in self.model._fc.parameters():
                param.requires_grad = True
        else:
            for param in self.model.fc.parameters():
                param.requires_grad = True

    def getOptimizer(self):
        return self.optimizers()

