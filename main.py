import torch, time
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import os,sys,random
from data_loader import Load_Data
from Train_Test_Eval import Evaluate, Train_and_Test
from models.Baselines import BGD
import numpy as np
import torch_geometric
import torch.optim.lr_scheduler as lr_scheduler
import argparse, json




parser = argparse.ArgumentParser(
                    prog='Data Augmentation for Pearson Correlation-Based Functional Connectivity',
                    description='Add Noise for Data Augmentation',
                    epilog='Nothing Now')

#data options
parser.add_argument('--data_name', type=str, default='ABIDE200') #ABIDE200,ADHD200
parser.add_argument('--node_size', type=int, default=200) 
parser.add_argument('--num_of_fold', type=int, default=5)

#training options
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--data_load_device', type=str, default='cuda')
parser.add_argument('--denoise',type=bool, default=True)
parser.add_argument('--fisher_transform',type=bool, default=True)
parser.add_argument('--training_strategy', type=str, default='sd') #sd  cd  bd
parser.add_argument('--predictor', type=str, default='BNT')
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_of_epoch', type=int, default=1)
parser.add_argument('--random_seeds', type=list, default=[i for i in range(5)])

#model hyperparameters  
parser.add_argument('--weight_decay', type=float, default=1e-5)
parser.add_argument('--thres', type=int, default=50)

#evaluate options
parser.add_argument('--test_interval', type=int, default=20)
parser.add_argument('--save_path', type=str, default='resultRecord/')
parser.add_argument('--save_model', type=bool, default=False)
parser.add_argument('--save_result', type=bool, default=True)

parser.add_argument('--compare', type=bool, default=True)

args = parser.parse_args() 



Pretrain_data = Load_Data.PretrainData(args.data_name,args,device=args.data_load_device)
Dataset = Load_Data.MyDatasetRaw(args.data_name,args,device=args.data_load_device)
args.spec=f'{args.training_strategy}_{str(args.denoise)}'
for seed in args.random_seeds:
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    dataset = Load_Data.DataSpliter(Dataset,args.num_of_fold)
    lossfunc = {'CE':nn.CrossEntropyLoss(),'MSE':nn.MSELoss()}
    Trainevaluator = Evaluate.Evaluator()
    Evalevaluator = Evaluate.Evaluator()
    Evalevaluator2= Evaluate.Evaluator()
    Testevaluator = Evaluate.Evaluator()
    Eval = False
    for fold in range(1,args.num_of_fold+1):
        t1 = time.perf_counter()
        trainloader, evalloader = dataset.GetTrain(Batchsize=args.batch_size,FoldIndex=fold-1,eval=Eval)
        testloader = dataset.GetTest(Batchsize=args.batch_size,FoldIndex=fold-1)
        pretrainloader = torch.utils.data.DataLoader(Pretrain_data,batch_size=64, shuffle=True)
        model = BGD(args.node_size, predictor_type = args.predictor)
        model.to(args.device)
        if args.predictor == 'BNT': 
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr/10, weight_decay=args.weight_decay)
        else:
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optpre = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        e,repeat = 0,0
        while e < args.num_of_epoch+1:
            stop = Train_and_Test.train(trainloader,evalloader,pretrainloader,model,optimizer,optpre,lossfunc,e,Trainevaluator,Evalevaluator,args)
            if (e)%args.test_interval==0 or (e)==args.num_of_epoch:
                Train_and_Test.test(testloader,model,Testevaluator,Evalevaluator2,e,fold,args)
                if Eval:
                    print('epo:', e, "test acc:", round(Testevaluator.accuracy(),4), "val acc:", round(Evalevaluator.accuracy(),4),\
                    'train acc:', round(Trainevaluator.accuracy(),4), 'auc',round(Testevaluator.AUC(),4),'loss', \
                        round(Trainevaluator.losses[0],4),round(Trainevaluator.losses[1],4),round(Trainevaluator.losses[2],4))
                else:
                    print('epo:', e, "test acc:", round(Testevaluator.accuracy(),4),'auc:',Testevaluator.AUC(),'pre:',Testevaluator.precision(),\
                        'rec:',Testevaluator.recall(), 'sen:', Testevaluator.sensitivity(), 'spe:', Testevaluator.specificity(),\
                            'train acc:', round(Trainevaluator.accuracy(),4),'loss:', round(Trainevaluator.losses[0],4),\
                                round(Trainevaluator.losses[1],4),round(Trainevaluator.losses[2],4))
            if e==100 and Trainevaluator.accuracy()<0.9999999 and repeat<=3:  
                e-=10
                repeat+=1
            e+=1
        Testevaluator.accuracy(record=True)
        Testevaluator.recall(record=True)
        Testevaluator.precision(record=True)
        Testevaluator.AUC(record=True)
        Testevaluator.sensitivity(record=True)
        Testevaluator.specificity(record=True)
        if args.save_model:
            Train_and_Test.save(fold,model,Testevaluator,seed, args,'model')#save result for each fold
        print('seed:',seed,'fold:',fold+1,'mean:',Testevaluator.foldAVG('acc'),'all_acc',Testevaluator.record['acc'],'auc:',Testevaluator.foldAVG('auc'), \
            'specs:',args.predictor, args.spec,)
        t2 = time.perf_counter()
        print(f'time comsumed: {t2-t1}\n')
    Train_and_Test.save(fold,model,Testevaluator,seed,args,'metric') #saving result for each seed
Evaluate.compare(args) # writing all result in one csv