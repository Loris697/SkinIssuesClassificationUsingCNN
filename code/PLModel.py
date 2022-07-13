import pytorch_lightning as pl
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
import madgrad
import torch
import numpy as np
from torchvision import transforms
import sys  
sys.path.insert(0, './code')
from RADAM import RAdam

import logging
import warnings

logging.getLogger("lightning").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


class PLModel(pl.LightningModule):
    def __init__(self, name, model, train_dataset = None , val_dataset = None, batch_size = 512, loss = None, lastLayers=None, num_epochs = 0, path = None, lr = 0.001, wd = 0.0, reduced = None, optimName = 'RADAM', BYOL = False, aug = None):
        super(PLModel, self).__init__()

        self.maxTestAcc = 0
        self.optimName = optimName
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size
        self.maxValAcc = 0
        self.testAcc = 0
        self.lr = lr
        self.bestModel = None
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
            
        if aug is None:
            self.aug = nn.Sequential(
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.05, hue=0.05),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomAffine((-180, 180), fill=0, scale = (0.7, 1.7), shear=(-30, 30))
            )
        else:
            self.aug = aug
                  
            
        if path is None:
            self.PATH = 'skin/supervised/'+self.type+self.reduced+self.optimName+self.name+'.ckpt'
            
        if name.find('EfficientNet') > -1:
            self.numFeat = self.model._fc.in_features
        elif name.find('DenseNet') > -1:
            self.numFeat = self.model.classifier.in_features
        else:
            self.numFeat = self.model.fc.in_features
        
        if lastLayers is not None:
            if name.find('EfficientNet') > -1:
                numFeat = self.model._fc.in_features
                self.model._fc = nn.Linear( numFeat, numFeat)
            else:
                numFeat = self.model.fc.in_features
                self.model.fc = nn.Linear( numFeat, numFeat)
            self.isLL = lastLayers

        self.writer = self.type+self.reduced+self.optimName+ self.name
        
        
        
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
        sched1 = {'scheduler': scheduler, 'monitor': 'valLoss'}
        #print(optimizer)
        return {
           'optimizer': optimizer,
           'lr_scheduler': sched1
       }
    
    def training_step(self, batch, batch_idx):
        acc = 0
        x, y = batch
        
        with torch.no_grad():
            x = self.aug(x)
        
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
        
        self.log('trainAcc', avg_train_acc, prog_bar=False)
        self.log('trainLoss', avg_train_loss, prog_bar=False)
    
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

        for i in range(len(self.labelName)):
            if n_class_samples[i] != 0:
                bal_acc += n_class_correct[i]/n_class_samples[i]
                self.log('valAcc'+self.labelName[i], n_class_correct[i]/n_class_samples[i], prog_bar=False)

        bal_acc = bal_acc/len(self.labelName)

        self.log('valAcc', avg_val_acc, prog_bar=False)
        self.log('valLoss', avg_val_loss, prog_bar=True)
        self.log('balValAcc', bal_acc, prog_bar=True)
        
        
        if bal_acc > self.maxValAcc:
            self.maxValAcc = avg_val_acc
            self.bestModel = self.model.state_dict()
                
    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
        
            dataset=self.train_dataset, batch_size=self.batch_size, num_workers=4, shuffle=True
        )
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            dataset=self.val_dataset, batch_size=self.batch_size, num_workers=4, shuffle=False
        )
        return val_loader
    
    def recoverBestModel(self):
        print("Recovering best model")
        self.model.load_state_dict(self.bestModel)

    def fineTuning(self, layerToUnlock):
        if layerToUnlock > 4:
            print("I livelli da bloccare possono essere al massimo 4")
            return False
        for param in self.model.parameters():
            param.requires_grad = False
        if self.name.find('EfficientNet') > -1:
            for param in self.model._fc.parameters():
                param.requires_grad = True
            for i in range(len(self.model._blocks)):
                if i > (len(self.model._blocks) - layerToUnlock*7):
                    for param in self.model._blocks[i].parameters():
                        param.requires_grad = True
        elif self.name.find('DenseNet') > -1:
            for param in self.model.classifier.parameters():
                param.requires_grad = True
            if layerToUnlock > 3:
                for param in self.model.features.denseblock1.parameters():
                    param.requires_grad = True
            if layerToUnlock > 2:
                for param in self.model.features.denseblock2.parameters():
                    param.requires_grad = True
            if layerToUnlock > 1:
                for param in self.model.features.denseblock3.parameters():
                    param.requires_grad = True
            if layerToUnlock > 0:
                for param in self.model.features.denseblock4.parameters():
                    param.requires_grad = True
            
        else:
            for param in self.model.fc.parameters():
                param.requires_grad = True
            if layerToUnlock > 3:
                for param in self.model.layer1.parameters():
                    param.requires_grad = True
            if layerToUnlock > 2:
                for param in self.model.layer2.parameters():
                    param.requires_grad = True
            if layerToUnlock > 1:
                for param in self.model.layer3.parameters():
                    param.requires_grad = True
            if layerToUnlock > 0:
                for param in self.model.layer4.parameters():
                    param.requires_grad = True
