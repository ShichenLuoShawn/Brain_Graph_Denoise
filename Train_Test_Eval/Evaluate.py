import torch
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
import pandas as pd
import os

class Evaluator():
    def __init__(self):
        self.fold = 1
        self.record = {'acc':[],'pre':[],'rec':[],'sen':[],'spe':[],'auc':[]} # {epoch:acc}
        self.temptp = 0
        self.tempfp = 0
        self.temptn = 0
        self.tempfn = 0
        self.loss = 0
        self.losses=[]
        self.preds = np.array([])
        self.labels = np.array([])

    def update(self, out, groundtruth,loss):
        self.labels = np.concatenate([self.labels,groundtruth.flatten().to('cpu')],0)
        self.preds = np.concatenate([self.preds,F.softmax(out,-1)[:,1].flatten().detach().to('cpu')],0)
        pred = torch.argmax(out,-1)
        self.temptp += ((pred==1)*(groundtruth==1)).sum() #true positive
        self.tempfp += ((pred==1)*(groundtruth==0)).sum()
        self.temptn += ((pred==0)*(groundtruth==0)).sum()
        self.tempfn += ((pred==0)*(groundtruth==1)).sum()
        self.losses = loss
        if loss:
            self.loss = torch.tensor(loss).sum().item()

    def accuracy(self,record=False):
        acc = round(((self.temptp+self.temptn)/(self.temptp+self.tempfp+self.temptn+self.tempfn)).item(),4)
        if record:
            self.record['acc'].append(acc)
        return acc
    
    def precision(self,record=False):
        pre = round((self.temptp/(self.temptp+self.tempfp)).item(),4)
        if record:
            self.record['pre'].append(pre)
        return pre
    
    def recall(self,record=False):
        rec = round((self.temptp/(self.temptp+self.tempfn)).item(),4)
        if record:
            self.record['rec'].append(rec)
        return rec
    
    def sensitivity(self,record=False):
        sen = round((self.temptp/(self.temptp+self.tempfn)).item(),4)
        if record:
            self.record['sen'].append(sen)
        return sen
    
    def specificity(self,record=False):
        spe = round((self.temptn/(self.temptn+self.tempfp)).item(),4)
        if record:
            self.record['spe'].append(spe)
        return spe
    
    def AUC(self,record=False):
        try:
            auc = round(metrics.roc_auc_score(self.labels,self.preds),4)
        except:
            auc = 0
        if record:
            self.record['auc'].append(auc)
        return auc
    
    def loss(self,record=False):
        if record:
            self.record['loss'].append(self.losses)
        return self.losses
    
    def foldAVG(self,metrics):
        return round((torch.tensor(self.record[metrics]).float()).mean().item(),4)


    def reset(self, fold):
        self.temptp = 0
        self.tempfp = 0
        self.temptn = 0
        self.tempfn = 0
        self.loss=0
        self.fold = fold
        self.preds = np.array([])
        self.labels = np.array([])
    
    def save_result(self, path_name):
        for key in self.record.keys():
            data = self.record[key]
            mean = np.array(data).mean()
            std = np.array(data).std()
            self.record[key].append(mean)
            self.record[key].append(std)
        data = pd.DataFrame(self.record)
        data.index = [1,2,3,4,5,'mean','std']
        data.to_csv(f'{path_name}.csv')

    def save_result_LOSO(self, path_name, length,site):
        for key in self.record.keys():
            data = self.record[key]
            mean = np.array(data).mean()
            std = np.array(data).std()
            self.record[key].append(mean)
            self.record[key].append(std)
            self.record[key].append(length)
            self.record[key].append(site)
        data = pd.DataFrame(self.record)
        data.index = [1,'mean','std','length','site']
        data.to_csv(f'{path_name}.csv')

def compare(args):
    record = {'acc':[],'pre':[],'rec':[],'sen':[],'spe':[],'auc':[]}
    for seed in args.random_seed:
        record_file_name = f'data={args.data_name}_model={args.predictor}_seed={seed}.csv'
        subpath = f'./resultRecord/data={args.data_name}_model={args.predictor}_{args.spec}'
        path = os.path.join(subpath,record_file_name)
        res = pd.read_csv(path)
        res = res.set_index('Unnamed: 0')
        mean = res.loc['mean']
        for key in record.keys():
            record[key].append(mean[key])
    for key in record.keys():
        record[key].append(np.array(record[key]).mean())
    data = pd.DataFrame(record)
    data.to_csv(f'{subpath}/result.csv')