import argparse
import sys
import os
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
import yaml
from data_utils_T3 import genSpoof_list, Dataset_for, Dataset_for_eval
from mchad import MCHAD
# from model_phucdt import Model
from tensorboardX import SummaryWriter
from core_scripts.startup_config import set_random_seed


__author__ = "Hemlata Tak"
__email__ = "tak@eurecom.fr"



def evaluate_accuracy(dev_loader, model, device):
    val_loss = 0.0
    num_total = 0.0
    model.eval()
    # weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss()
    num_correct = 0.0
    for batch_x, batch_fs, batch_y in dev_loader:
        
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_x = batch_x.to(device)
        batch_fs = batch_fs.to(device)
        
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x, batch_fs)
        
        _, batch_y_hat, _ = model.step(batch_out, batch_y)
        num_correct += (batch_y_hat == batch_y).sum(dim=0).item()
   
    return 100 * (num_correct / num_total)

def extract_embedding(
    dataset,
    model,
    device,
    save_path):
    """Perform evaluation and save the score to a file"""
    data_loader = DataLoader(dataset, batch_size=32, shuffle=False, drop_last=False)
    model.eval()
    
    for batch_x, batch_fs, utt_id in data_loader:

        batch_size = batch_x.size(0)
        batch_x = batch_x.to(device)
        batch_fs = batch_fs.to(device)
        batch_out = model(batch_x, batch_fs)

        for id, sco in zip(utt_id, batch_out.tolist()):
            torch.save(sco, save_path + "/" + id)

    print("bio saved to {}".format(save_path))

def produce_evaluation_file(dataset, model, device, save_path):
    data_loader = DataLoader(dataset, batch_size=128, shuffle=False, drop_last=False)
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    
    fname_list = []
    key_list = []
    score_list = []
    emb_list = []
    
    for batch_x, batch_fs, utt_id in data_loader:
        fname_list = []
        score_list = []  
        batch_size = batch_x.size(0)
        batch_x = batch_x.to(device)
        batch_fs = batch_fs.to(device)
        
        _, batch_out_y = model.predict(batch_x, batch_fs)

        # add outputs
        fname_list.extend(utt_id)
        score_list.extend(batch_out_y.detach().cpu().tolist())
        # emb_list.extend(batch_out_z.detach().cpu().tolist())
        
        with open(save_path, 'a+') as fh:
            for f, cm in zip(fname_list,score_list):
                fh.write('{} {}\n'.format(f, cm))
        fh.close()   
    print('Scores saved to {}'.format(save_path))



def train_epoch(train_loader, model, lr,optim, device):
    running_loss = 0
    
    num_total = 0.0
    
    model.train()

    #set objective (Loss) functions
    weight = torch.FloatTensor([0.1, 0.9]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    
    for batch_x, batch_fs, batch_y in train_loader:
       
        batch_size = batch_x.size(0)
        num_total += batch_size
        
        batch_x = batch_x.to(device)
        batch_fs = batch_fs.to(device)
        batch_y = batch_y.view(-1).type(torch.int64).to(device)
        batch_out = model(batch_x, batch_fs)
        
        batch_loss, _,_ = model.step(batch_out, batch_y)
        
        running_loss += (batch_loss.item() * batch_size)
       
        optimizer.zero_grad()
        batch_loss.backward()
        optimizer.step()
       
    running_loss /= num_total
    
    return running_loss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASVspoof2021 baseline system')
    # Dataset
    parser.add_argument('--train_path', type=str, default='/your/path/to/data/', help='train set')
    parser.add_argument('--dev_path', type=str, default='/your/path/to/data/', help='dev set')
    parser.add_argument('--database_path', type=str, default='/your/path/to/data/', help='eval set')
    '''
    % database_path/
    %   | - protocol.txt
    %   | - audio_path
    '''

    parser.add_argument('--protocols_path', type=str, default='database/', help='Change with path to user\'s LA database protocols directory address')
    '''
    % database_path/
    %   | - protocol.txt
    %   | - audio_path

    '''

    # Hyperparameters
    parser.add_argument('--batch_size', type=int, default=14)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.000001)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--loss', type=str, default='weighted_CCE')
    # model
    parser.add_argument('--seed', type=int, default=1234, 
                        help='random seed (default: 1234)')
    
    parser.add_argument('--model_path', type=str,
                        default=None, help='Model checkpoint')
    parser.add_argument('--comment', type=str, default=None,
                        help='Comment to describe the saved model')
    # Auxiliary arguments
    parser.add_argument('--track', type=str, default='LA',choices=['LA', 'PA','DF'], help='LA/PA/DF')
    parser.add_argument('--eval_output', type=str, default=None,
                        help='Path to save the evaluation result')
    parser.add_argument('--eval', action='store_true', default=False,
                        help='eval mode')
    parser.add_argument('--is_eval', action='store_true', default=False,help='eval database')
    parser.add_argument('--extract_emb', action='store_true', default=False,help='Extract embeddings of database_path to eval_output directory')
    parser.add_argument('--eval_part', type=int, default=0)
    # backend options
    parser.add_argument('--cudnn-deterministic-toggle', action='store_false', \
                        default=True, 
                        help='use cudnn-deterministic? (default true)')    
    
    parser.add_argument('--cudnn-benchmark-toggle', action='store_true', \
                        default=False, 
                        help='use cudnn-benchmark? (default false)') 


    ##===================================================Rawboost data augmentation ======================================================================#

    parser.add_argument('--algo', type=int, default=5, 
                    help='Rawboost algos discriptions. 0: No augmentation 1: LnL_convolutive_noise, 2: ISD_additive_noise, 3: SSI_additive_noise, 4: series algo (1+2+3), \
                          5: series algo (1+2), 6: series algo (1+3), 7: series algo(2+3), 8: parallel algo(1,2) .[default=0]')

    # LnL_convolutive_noise parameters 
    parser.add_argument('--nBands', type=int, default=5, 
                    help='number of notch filters.The higher the number of bands, the more aggresive the distortions is.[default=5]')
    parser.add_argument('--minF', type=int, default=20, 
                    help='minimum centre frequency [Hz] of notch filter.[default=20] ')
    parser.add_argument('--maxF', type=int, default=8000, 
                    help='maximum centre frequency [Hz] (<sr/2)  of notch filter.[default=8000]')
    parser.add_argument('--minBW', type=int, default=100, 
                    help='minimum width [Hz] of filter.[default=100] ')
    parser.add_argument('--maxBW', type=int, default=1000, 
                    help='maximum width [Hz] of filter.[default=1000] ')
    parser.add_argument('--minCoeff', type=int, default=10, 
                    help='minimum filter coefficients. More the filter coefficients more ideal the filter slope.[default=10]')
    parser.add_argument('--maxCoeff', type=int, default=100, 
                    help='maximum filter coefficients. More the filter coefficients more ideal the filter slope.[default=100]')
    parser.add_argument('--minG', type=int, default=0, 
                    help='minimum gain factor of linear component.[default=0]')
    parser.add_argument('--maxG', type=int, default=0, 
                    help='maximum gain factor of linear component.[default=0]')
    parser.add_argument('--minBiasLinNonLin', type=int, default=5, 
                    help=' minimum gain difference between linear and non-linear components.[default=5]')
    parser.add_argument('--maxBiasLinNonLin', type=int, default=20, 
                    help=' maximum gain difference between linear and non-linear components.[default=20]')
    parser.add_argument('--N_f', type=int, default=5, 
                    help='order of the (non-)linearity where N_f=1 refers only to linear components.[default=5]')

    # ISD_additive_noise parameters
    parser.add_argument('--P', type=int, default=10, 
                    help='Maximum number of uniformly distributed samples in [%].[defaul=10]')
    parser.add_argument('--g_sd', type=int, default=2, 
                    help='gain parameters > 0. [default=2]')

    # SSI_additive_noise parameters
    parser.add_argument('--SNRmin', type=int, default=10, 
                    help='Minimum SNR value for coloured additive noise.[defaul=10]')
    parser.add_argument('--SNRmax', type=int, default=40, 
                    help='Maximum SNR value for coloured additive noise.[defaul=40]')
    
    ##===================================================Rawboost data augmentation ======================================================================#
    

    if not os.path.exists('models_T3'):
        os.mkdir('models_T3')
    if not os.path.exists('logs_T3'):
        os.mkdir('logs_T3')
    args = parser.parse_args()
 
    #make experiment reproducible
    set_random_seed(args.seed, args)
    
    # track = args.track

    # assert track in ['LA', 'PA','DF'], 'Invalid track given'

    # #database
    # prefix      = 'ASVspoof_{}'.format(track)
    # prefix_2019 = 'ASVspoof2019.{}'.format(track)
    # prefix_2021 = 'ASVspoof2021.{}'.format(track)
    
    
    # #define model saving path
    model_tag = 'model_{}_{}_{}_{}'.format(
        args.loss, args.num_epochs, args.batch_size, args.lr)
    if args.comment:
        model_tag = model_tag + '_{}'.format(args.comment)
    model_save_path = os.path.join('models_T3', model_tag)

    #set model save directory
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    
    #GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'                  
    print('Device: {}'.format(device))
    
    model = MCHAD(args, device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    model =model.to(device)
    print('nb_params:',nb_params)

    #set Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)
    
    if args.model_path:
        model.load_state_dict(torch.load(args.model_path,map_location=device))
        print('Model loaded : {}'.format(args.model_path))

    #extract embeddings
    if args.extract_emb:
        file_eval = genSpoof_list( dir_meta =  os.path.join(args.database_path+'/protocol.txt'),is_train=False,is_eval=True)
        print('no. of eval trials',len(file_eval))
        eval_set=Dataset_for_eval(list_IDs = file_eval,base_dir = os.path.join(args.database_path+'/'))
        extract_embedding(eval_set, model, device, args.eval_output)
        sys.exit(0)

    #evaluation 
    if args.eval:
        file_eval = genSpoof_list( dir_meta =  os.path.join(args.database_path+'/protocol.txt'),is_train=False,is_eval=True)
        print('no. of eval trials',len(file_eval))
        eval_set=Dataset_for_eval(list_IDs = file_eval,base_dir = os.path.join(args.database_path+'/'))
        produce_evaluation_file(eval_set, model, device, args.eval_output)
        sys.exit(0)
   
     
    # define train dataloader
    d_label_trn,file_train = genSpoof_list( dir_meta =  os.path.join(args.train_path+'protocol.txt'),is_train=True,is_eval=False)
    
    print('no. of training trials',len(file_train))
    
    train_set=Dataset_for(args,list_IDs = file_train,labels = d_label_trn,base_dir = args.train_path+'/',algo=args.algo)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size,num_workers=8, shuffle=True,drop_last = True)
    
    del train_set,d_label_trn
    

    # define validation dataloader

    d_label_dev,file_dev = genSpoof_list(dir_meta = args.dev_path+'protocol.txt',is_train=False,is_eval=False)
    
    print('no. of validation trials',len(file_dev))
    
    dev_set = Dataset_for(args,list_IDs = file_dev,
		labels = d_label_dev,
		base_dir = args.dev_path+'/',algo=args.algo)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size,num_workers=8, shuffle=False)
    del dev_set,d_label_dev

    
    

    # Training and validation 
    num_epochs = args.num_epochs
    writer = SummaryWriter('logs_T3/{}'.format(model_tag))
    
    for epoch in range(num_epochs):
        
        running_loss = train_epoch(train_loader,model, args.lr,optimizer, device)
        val_loss = evaluate_accuracy(dev_loader, model, device)
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('loss', running_loss, epoch)
        print('\n{} - {} - {} '.format(epoch,
                                                   running_loss,val_loss))
        torch.save(model.state_dict(), os.path.join(
            model_save_path, 'epoch_{}.pth'.format(epoch)))
