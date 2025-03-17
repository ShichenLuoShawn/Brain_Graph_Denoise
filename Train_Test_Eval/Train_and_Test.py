import torch, math
from Train_Test_Eval import Evaluate
import sys, os, json, argparse, random
import torch.nn.functional as F
import pandas as pd
from matplotlib import pyplot as plt

zero_replace = torch.tensor(0,device='cuda')

def train(dataloader,evalloader,pretrainloader,model, optimizer,optpre, lossfunc, epoch, trainevaluator, subevaluator\
         ,args):
    model.train()
    trainevaluator.reset(epoch)
    record = {'train_loss':[],'test_loss':[],'original_loss':[],'lengthdif':[]}
    if args.denoise:
        L_diff,L_diff_train = [],[]
        pretrainmaxepoch = 300 if epoch < 1 else 0
        for pretrain_epo in range(pretrainmaxepoch):
            trloss,tloss,oloss=0,0,0
            avgloss,count=0,0
            for i, (target_corr,ori_ts,orilengths) in enumerate(pretrainloader):
                target_corr,ori_ts,orilengths = target_corr.to(args.device), ori_ts.to(args.device), orilengths.to(args.device)
                source_ts, source_length = subsequence_sample(ori_ts,orilengths,thres=args.thres)
                norm_source_ts = (source_ts-source_ts.mean(-1,keepdim=True))/(source_ts.std(-1,keepdim=True)+1e-9)
                source_corr = (norm_source_ts@norm_source_ts.transpose(2,1))/source_length[0]

                if args.fisher_transform:
                    target_corr = 1/2*torch.log((1+target_corr+1e-9)/(1-target_corr+1e-9))
                target_corr = target_corr/target_corr[:,0,0][:,None,None]
                source_corr_denoised = model(source_corr,source_length,'train',pretrain=True,epoch=pretrain_epo,args=args)
                loss = ((target_corr-source_corr_denoised)**2).mean()
                optpre.zero_grad()
                loss.backward()
                optpre.step()
                avgloss+=loss
                count+=1

            if (pretrain_epo)%100==0:
                print('Pretraining Loss:',round((avgloss/count*10).item(),4))

    for i,(inptrain,label,length) in enumerate(dataloader):
        inp, label, length = inptrain.to(args.device), label.long().to(args.device), length.to(args.device)
        out = model(inp,length,'train',pretrain=False,epoch=epoch,args=args)
        CL = lossfunc['CE'](torch.Tensor(out), label)
        trainevaluator.update(out,label,[CL.item(),zero_replace.item(),zero_replace.item()])#genloss.item(),disloss.item()])
        optimizer.zero_grad()
        CL.backward()
        optimizer.step()
    return False
 

def test(dataloader, model, testevaluator, subevaluator, epoch, fold, args):
    model.eval()
    testevaluator.reset(0)
    for i,(inptest,label,length) in enumerate(dataloader):
        inp, label, length = inptest.to(args.device), label.to(args.device), length.to(args.device)
        targetLength = torch.ones(length.shape,device='cuda')*args.thres
        out = model(inp,length,'test',pretrain=False,epoch=epoch,args=args)
        testevaluator.update(out,label,0)

def subsequence_sample(ts,t_length,thres=200):
    min_length = t_length.min()
    target_length=min_length
    random_length = random.randint(50,target_length-thres)
    random_start = random.randint(0,target_length-random_length)
    source_ts = ts[:,:,random_start:random_start+random_length]
    return source_ts, torch.tensor(random_length,device='cuda').repeat(source_ts.shape[0])

def save(fold, model, testevaluator, seed, args, content='model'):
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    record_file_name = f'data={args.data_name}_model={args.predictor}_seed={seed}'
    subpath = f'data={args.data_name}_model={args.predictor}_{args.spec}'
    subpath = os.path.join(args.save_path, subpath)
    if not os.path.isdir(subpath):
        os.makedirs(subpath)
    finalpath = os.path.join(subpath, record_file_name)
    if content == 'model':
        print('saving model......')
        torch.save(model.state_dict(),finalpath)
        with open(f'{subpath}/args.txt', 'w') as f:
            json.dump(args.__dict__, f, indent=2)
    elif content == 'metric':
        print('saving metrics......')
        testevaluator.save_result(finalpath)

def load_args(path):
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    with open(path, 'r') as f:
        args.__dict__ = json.load(f)


