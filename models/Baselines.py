import torch
import torch.nn as nn
import torch.nn.functional as F
import math,sys
import matplotlib.pyplot as plt
import torch_geometric.nn as gnn
import torch_geometric.utils as utils
import pandas as pd
from models.braingnn.braingnn import Network
from models.BNT.BNT.bnt import BrainNetworkTransformer

class AE(nn.Module):
    def __init__(self,num_of_node=200):
        super().__init__()
        hidden=128
        self.enc1 = nn.Linear(num_of_node,hidden,bias=True)
        self.enc2 = nn.Linear(num_of_node,hidden,bias=True)
        self.dec2 = nn.Linear(hidden,num_of_node,bias=True)
        self.dec1 = nn.Linear(hidden,num_of_node,bias=True)
        self.act = nn.ReLU()

    def forward(self,x,reminder=None): 
        out1 = self.act((self.enc1(x)).transpose(2,1))
        out2 = self.act(self.enc2(out1))
        out2 = self.act((self.dec2(out2))+out1).transpose(2,1)
        out2 = self.dec1(out2)
        return out2


class DA(nn.Module):
    def __init__(self,num_of_node=200):
        super().__init__()
        self.num_of_node = num_of_node
        self.act = nn.ReLU()
        self.AE = AE(num_of_node)
    
    def forward(self, x, slength, orix, epoch=0):
        deltax = self.AE(x)
        return (deltax+deltax.transpose(2,1))/2

class predictor(nn.Module):
    def __init__(self, in_dim=200,hidden=64,ModelName='GCN',args=None):
        super().__init__()
        self.ModelName = ModelName
        self.num_nodes = in_dim
        if ModelName == 'GAT':
            self.embedding = gnn.Sequential('x, edge_index,attr',[(gnn.GATConv(in_dim,128),'x, edge_index,attr -> x'),
                                                             nn.ReLU(inplace=True),
                                                             (gnn.GATConv(128,128),'x, edge_index,attr -> x')])
        elif ModelName == 'GIN':
            self.embedding = gnn.Sequential('x, edge_index',[(gnn.GINConv(nn.Linear(in_dim,128)),'x, edge_index -> x'),
                                                            nn.ReLU(inplace=True),
                                                            (gnn.GINConv(nn.Linear(128,128)),'x, edge_index -> x')])
        elif ModelName == 'GCN':
            self.embedding = gnn.Sequential('x, edge_index,attr',[(gnn.GCNConv(in_dim,128),'x, edge_index, attr -> x'),
                                                            nn.ReLU(inplace=True),
                                                            (gnn.GCNConv(128,128),'x, edge_index, attr -> x')])
        elif ModelName == 'AE':
            flat_in_dim = int((in_dim*in_dim-in_dim)/2)
            self.embedding = nn.Sequential(nn.Linear(flat_in_dim,512),nn.ReLU(),nn.Linear(512,128))
        
        elif ModelName == 'BNT':
            self.embedding = BrainNetworkTransformer(in_dim)
        
        # elif ModelName == 'IBGNN':
        #     self.embedding = build_model(in_dim)
        
        elif ModelName == 'BrainGNN':
            self.embedding = Network(in_dim,0.5,2,8,in_dim)

    def forward(self,x,adj):
        shape = x.shape
        adj = sparse(x,0.2)
        if self.ModelName[0] == 'G' or self.ModelName == 'BrainGNN':
            edge_index,edge_attr = utils.dense_to_sparse(adj)
            x = x.reshape(shape[0]*shape[1],shape[2])
            if self.ModelName == 'BrainGNN':
                batch = torch.tensor([i//x.shape[-1] for i in range(x.shape[0])],device='cuda')
                pos = torch.eye(shape[-1],shape[-1],device='cuda').repeat(shape[0],1,1)
                x = self.embedding(x,edge_index,batch,edge_attr,pos)
            else:
                x = self.embedding(x,edge_index,edge_attr)
                x = x.reshape(shape[0],shape[1],-1).mean(-2)
        elif self.ModelName == 'AE':
            x[x==0] = 1e-9
            x = torch.tril(x,-1)
            x = x[x!=0]
            x = x.reshape(shape[0],-1)
            x = self.embedding(x)
        elif self.ModelName == 'BNT':
            x = self.embedding(x)
        return x


class BGD(nn.Module):
    def __init__(self,in_dim=200, hidden1=600, hidden2=200, predictor_type='GCN'):
        super().__init__()
        self.predictor = predictor(in_dim,ModelName=predictor_type)
        self.fcpout = nn.Sequential(nn.Linear(128,2))
        self.act = nn.ReLU()
        self.DAmodule = DA(in_dim)
        self.remove_eye = 1-torch.eye(in_dim,device='cuda')
    
    def forward(self, x, slength, mode='train', pretrain=False, epoch=0, args=None):
        adj = x.clone().detach()
        if args.fisher_transform:
            x = 1/2*torch.log((1+x+1e-9)/(1-x+1e-9))
        x = x/x[:,0,0][:,None,None]
        xori = x.detach()
        if args.denoise:
            deltax = self.DAmodule(x,slength,adj,epoch=epoch)
            x = x+deltax*self.remove_eye
            if pretrain:
                return x
        out = self.predictor(x.detach(),adj)
        out = self.act(out)
        out = self.fcpout(out)
        return out
    
    
def sparse(x,rate):
    shape = x.shape
    value,index = torch.topk(x.reshape(shape[0],-1),int(shape[-1]*shape[-2]*rate),-1)
    mask = torch.zeros(x.shape,device='cuda').reshape(shape[0],-1)
    mask[torch.arange(shape[0])[:,None],index] = 1
    mask = mask.reshape(shape[0],shape[1],shape[2])
    return x*mask