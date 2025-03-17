from torch.utils.data import Dataset
from torch.utils.data import dataloader
import torch
import torch.nn.functional as F
from scipy.io import loadmat,savemat
import numpy as np
import pandas as pd
import os, sys
import matplotlib.pyplot as plt
import seaborn as sn
import random, re
from matplotlib import pyplot as plt
from collections import Counter

remain_index = [i for i in range(200) if i not in [1,4,10,24,33,67,81,94,125,190]]

class MyDatasetRaw(Dataset):
    def __init__(self,dataname,args,device='cpu'):
        self.device=device
        self.thres = args.thres
        self.corr = []
        self.labels = []
        self.length = []
        self.siteID = []
        self.path = f'..\\Data\\{dataname}'
        if dataname == 'ADHD200':
            self.metadata = pd.read_csv(f'..\\Data\\{dataname}\\'+'label.csv')[["subject","DX_GROUP",'subindex','site']]
        else:
            self.metadata = pd.read_csv(f'..\\Data\\{dataname}\\'+'label.csv')[["subject","DX_GROUP",'SITE_ID']]
        self.LoadData()
        length = np.array(self.length)
        self.ori_ts=[]
        self.orilengths=[]
        
    def __len__(self):
        return len(self.corr)
    
    def __getitem__(self, index):
        return self.corr[index],self.labels[index],self.length[index]#,self.siteID[index]

    def LoadData(self):
        for root, dirs, files in os.walk(self.path):
            for file in files:
                if file.endswith('cc200.1D'):
                    timeSeries = torch.Tensor(np.loadtxt(root+'\\'+file,dtype=float).transpose(1,0)).cuda()#[remain_index,:]#.to(self.device)
                    ID = file[-19:-14]
                    label = self.metadata.loc[self.metadata['subject']==int(ID)].iloc[0,1]-1 # 1 autism 0 control
                    site = self.metadata.loc[self.metadata['subject']==int(ID)].iloc[0,2]

                if file.endswith('cc200.csv'):
                    ID = int(re.search('^[0-9]*_',file).group()[0:-1])
                    label = self.metadata.loc[self.metadata['subject']==int(ID)].iloc[0,1]
                    if label == 'withheld': continue
                    else: label = int(label) 
                    site = self.metadata.loc[self.metadata['subject']==int(ID)].iloc[0,3]
                    label = 1 if label>=1 else 0
                    timeSeries = torch.Tensor(np.array(pd.read_csv(root+'\\'+file)).transpose(1,0))[1:,:]

                length = timeSeries.shape[1]
                timeSeries = (timeSeries - timeSeries.mean(-1,keepdim=True))/(timeSeries.std(-1,keepdim=True)+1e-9)
                corr = timeSeries@timeSeries.T/length

                self.corr.append(corr.cuda())
                self.labels.append(label)
                self.length.append(length)
                self.siteID.append(site)

        self.labels=torch.Tensor(self.labels).to(self.device)

class PretrainData(Dataset):
    def __init__(self,dataname,args,normal=True,device='cpu'):
        self.device=device
        self.min_target=150
        self.ori_ts=[]
        self.length = []
        self.corr = []
        if args.training_strategy == 'sd': #same datasets training
            self.path = [f'..\\Data\\{dataname}']
        elif args.training_strategy == 'bd': #both datasets training
            self.path = [f'..\\Data\\ADHD200',f'..\\Data\\ABIDE200']
        elif args.training_strategy == 'cd': #cross dataset training
            self.path = [i for i in [f'..\\Data\\ADHD200',f'..\\Data\\ABIDE200'] if i!=f'..\\Data\\{dataname}']
        self.LoadData()
        length = torch.tensor(np.array(self.length)).to(device)
        
    def __len__(self):
        return len(self.corr)
    
    def __getitem__(self, index):
        return self.corr[index], self.ori_ts[index], self.length[index]

    def LoadData(self):
        for path in self.path:
            for root, dirs, files in os.walk(path):
                for file in files:
                    if file.endswith('cc200.1D'):
                        ID = file[-19:-14]
                        timeSeries = torch.Tensor(np.loadtxt(root+'\\'+file,dtype=float).transpose(1,0)).cuda()
                    if file.endswith('cc200.csv'):
                        ID = int(re.search('^[0-9]*_',file).group()[0:-1])
                        timeSeries = torch.Tensor(np.array(pd.read_csv(root+'\\'+file)).transpose(1,0))[1:,:]
                    length = timeSeries.shape[1]
                    Norm_timeSeries = (timeSeries - timeSeries.mean(-1,keepdim=True))/(timeSeries.std(-1,keepdim=True)+1e-9)
                    corr = Norm_timeSeries@Norm_timeSeries.T/length
                    if (corr==0).sum()>0:continue
                    Norm_timeSeries = F.pad(Norm_timeSeries,(0,350-length))
                    if length>=self.min_target:
                        self.ori_ts.append(Norm_timeSeries.cuda())
                        self.length.append(length)
                        self.corr.append(corr.cuda())
    
class DataSpliter():
    def __init__(self,data,NumofFold:int,testRate=1,RSeed = 0):
        assert (testRate<=1 and testRate>0), 'testRate must within (0,1]'
        assert type(NumofFold)==int, 'NumofFold must be an integer'
        self.Rseed = RSeed
        self.data = data
        self.testRate = testRate
        self.NumofFold = NumofFold
        self.index, self.testsize = self.GetIndex()

    def GetTest(self, Batchsize, FoldIndex, shuffle=False):
        testindex = self.index[FoldIndex*self.testsize:(FoldIndex+1)*self.testsize]
        testset = torch.utils.data.Subset(self.data, testindex)
        return torch.utils.data.DataLoader(testset, batch_size=Batchsize, shuffle=shuffle)

    def GetTrain(self, Batchsize, FoldIndex, shuffle=True, eval=True):
        if not eval:
            trainindex = np.concatenate([self.index[:FoldIndex*self.testsize],self.index[(FoldIndex+1)*self.testsize:]])
            trainset = torch.utils.data.Subset(self.data, trainindex)
            return torch.utils.data.DataLoader(trainset, batch_size=Batchsize, shuffle=shuffle), 0
        else:
            trainindex = np.concatenate([self.index[:FoldIndex*self.testsize],self.index[(FoldIndex+1)*self.testsize:]])
            evalindex = trainindex[:self.testsize]
            trainindex = trainindex[self.testsize:]
            trainset = torch.utils.data.Subset(self.data, trainindex)
            evalset = torch.utils.data.Subset(self.data, evalindex)
            return torch.utils.data.DataLoader(trainset, batch_size=Batchsize, shuffle=shuffle), torch.utils.data.DataLoader(evalset, batch_size=Batchsize, shuffle=shuffle)

    def GetIndex(self):
        DataLength = len(self.data)
        index = np.random.permutation(np.array([i for i in range(DataLength)]))
        if self.testRate==1:
            testsize = DataLength//self.NumofFold
        else:
            testsize = int(DataLength//self.NumofFold*self.testRate)
        return index, testsize

if __name__=='__main__':
    #dataset = MyDataset('ABIDE.mat',False)
    #dataset=node_embedding('AAL90_region_info.xls')
    pass
    #for i in dataset:
    #    print(i)
    #    break
