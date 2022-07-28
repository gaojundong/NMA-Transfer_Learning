import os
import os.path as osp
import multiprocessing
import csv
from matplotlib import use
import pandas as pd
import time
import torch
import torch.nn as nn
from data_loader import HDF5Dataset,KDEFDataset
from torch.utils.data import DataLoader
from train_test import adjust_learning_rate
from ResNet import ResNet18, ResNet_pretrain_v1, ResNet_pretrain_v2
from train_test import train,test
from torchvision import transforms
import matplotlib.pyplot as plt
import gc



if torch.cuda.is_available():
    device=torch.device('cuda')
else:
    device = torch.device('cpu')

timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime()) 
use_cuda = torch.cuda.is_available()
    
torchvision_transforms = True  # True/False if you want use torchvision augmentations

start_epoch = 0
max_epochs = 300  # Please change this to 200
max_epochs_target = 300
base_learning_rate = 0.01
num_classes = 2
batch_size = 10 # try another bs
num_workers = 1  #multiprocessing.cpu_count()
#checkpointPath = '/home/gaojud96/DL_model/Transfer_Learning/checkpoint/18_FERG_scratch.t7' # model = ResNet18()
checkpointPath =  '/home/gaojud96/DL_model/Transfer_Learning/checkpoint/50_pretrain_v2_20220724_150536.t7' # model = ResNet_pretrain_v2()
outModelName = os.path.split(checkpointPath)[-1][:-3]+'_freezefc'
logname = '/home/gaojud96/DL_model/Transfer_Learning/results/' + 'transfer_bs32_lr01_e300_pt_do_n_' + outModelName + '.csv'
#outModelName = '18_FERG_v2_' + timestamp 



if __name__=='__main__':

    path = '/home/gaojud96/DL_model/Transfer_Learning/dataset'
    composed = transforms.Compose([
        # transforms.CenterCrop(224),
        # transforms.RandomInvert(p=0.1),
       #transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    transfer_traindataset = KDEFDataset(osp.join(path,'kdef_train_dataset.h5'),transform=composed)
    transfertrainloader = DataLoader(transfer_traindataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    transfer_testdataset = KDEFDataset(osp.join(path,'kdef_val_dataset.h5'),transform=composed)
    transfertestloader = DataLoader(transfer_testdataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    model = ResNet_pretrain_v2()
    #model = model.float()

      
    # # torch.cuda.empty_cache()
    # if os.path.isfile(checkpointPath):
    #     state_dict = torch.load(checkpointPath,map_location=device) #
    #     best_acc = state_dict['acc']
    #     print('Best Accuracy:', best_acc)
    #     if "state_dict" in state_dict:
    #         state_dict = state_dict["state_dict"]
    # # remove prefixe "module."
    #     state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    #     for k, v in model.state_dict().items():
    #         if k not in list(state_dict):
    #             print('key "{}" could not be found in provided state dict'.format(k))
    #         elif state_dict[k].shape != v.shape:
    #             print('key "{}" is of different shape in model and provided state dict'.format(k))
    #             state_dict[k] = v
    #     msg = model.load_state_dict(state_dict, strict=False)
    #     print("Load pretrained model with msg: {}".format(msg))
    # else:
    #     raise Exception('No pretrained weights found')
    

    # for param in model.parameters():
    #     param.requires_grad = True
    
    # if model = ResNet18() , frezze network with following code
    # num_ftrs = model.linear.in_features
    # model.linear = nn.Linear(num_ftrs, num_classes)

    # if model = ResNet_pretrain_v2() , use this code below to frezze network
    model.fc = nn.Sequential(
                nn.Linear(2048, 512),
                nn.Dropout(p=0.1, inplace=False),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.1, inplace=False),
                nn.Linear(512, 2)) 

    model = model.float()
    if use_cuda:
        #model.cuda()
        model.to(device)

    criterion = nn.CrossEntropyLoss()
    # if model = ResNet18() 
    # optimizer = torch.optim.SGD(model.linear.parameters(),lr=base_learning_rate, momentum=0.9,weight_decay=1e-4,)
    # if model =  ResNet_pretrain_v2()
    optimizer = torch.optim.SGD(model.parameters(),lr=base_learning_rate,momentum=0.9, weight_decay=0.0001)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Total Parameters:', total_params, 'Trainable parameters: ', trainable_total_params)


    if not os.path.exists(logname):
        with open(logname, 'w') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow(['epoch', 'train loss', 'train acc', 'test loss', 'test acc'])

    print('start training and testing')
    for epoch in range(start_epoch, max_epochs_target):

        adjust_learning_rate(optimizer, epoch)
        train_loss, train_acc = train(model, epoch,transfertrainloader,optimizer,criterion, use_cuda=use_cuda)
        test_loss, test_acc = test(model, epoch,transfertestloader, outModelName, criterion, use_cuda=use_cuda)
        with open(logname, 'a') as logfile:
            logwriter = csv.writer(logfile, delimiter=',')
            logwriter.writerow([epoch, train_loss, train_acc.item(), test_loss, test_acc.item()])
        print(f'Epoch: {epoch} | train acc: {train_acc} | test acc: {test_acc}')
    
