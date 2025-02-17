import argparse
import sys
import os
import numpy as np
import torch
from torch import nn
from torch import Tensor
from torch.utils.data import DataLoader
import yaml
from data_utils_fuse import genSpoof_list, genSpoof_list_custom, Dataset_for, Dataset_for_eval
from model import Model
# from model_phucdt import Model
from tensorboardX import SummaryWriter
from core_scripts.startup_config import set_random_seed
from tqdm import tqdm

__author__ = "Hemlata Tak"
__email__ = "tak@eurecom.fr"

class EarlyStop:
    def __init__(self, patience=5, delta=0, init_best=60, save_dir=''):
        self.patience = patience
        self.delta = delta
        self.best_score = init_best
        self.counter = 0
        self.early_stop = False
        self.save_dir = save_dir

    def __call__(self, score, model, epoch):
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            print("Best epoch: {}".format(epoch))
            self.best_score = score
            self.counter = 0
            # save model here
            torch.save(model.state_dict(), os.path.join(
                self.save_dir, 'epoch_{}.pth'.format(epoch)))

def evaluate_accuracy(dev_loader, model, device):
    val_loss = 0.0
    num_total = 0.0
    num_correct = 0.0
    model.eval()
    weight = torch.FloatTensor([0.19, 0.81]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    with torch.no_grad():
        for batch_x, batch_y in tqdm(dev_loader):
            
            batch_size = batch_x.size(0)
            num_total += batch_size
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)
            batch_out,_ = model(batch_x)
            _, batch_pred = batch_out.max(dim=1)
            num_correct += (batch_pred == batch_y).sum(dim=0).item()
            
            batch_loss = criterion(batch_out, batch_y)
            val_loss += (batch_loss.item() * batch_size)
            
        val_loss /= num_total
        val_accuracy = (num_correct/num_total)*100
   
    return val_loss, val_accuracy

def produce_emb_file(dataset, model, device, save_path, batch_size=10):
    data_loader = DataLoader(dataset, batch_size, shuffle=False, drop_last=False)
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    model.is_train = True

    fname_list = []
    key_list = []
    score_list = []
    with torch.no_grad():
        for batch_x, utt_id in tqdm(data_loader):
            fname_list = []
            score_list = []  
            pred_list = []
            batch_size = batch_x.size(0)
            batch_x = batch_x.to(device)
            
            batch_out, batch_emb = model(batch_x)
            score_list.extend(batch_out.data.cpu().numpy().tolist())
            # add outputs
            fname_list.extend(utt_id)

            # save_path now must be a directory
            # make dir if not exist
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # Then each emb should be save in a file with name is utt_id
            for f, emb in zip(fname_list,batch_emb):
                # normalize filename
                f = f.split('/')[-1].split('.')[0] # utt id only
                save_path_utt = os.path.join(save_path, f)
                np.save(save_path_utt, emb.data.cpu().numpy())
            
            # score file save into a single file
            with open(os.path.join(save_path, "scores.txt"), 'a+') as fh:
                for f, cm in zip(fname_list,score_list):
                    fh.write('{} {} {}\n'.format(f, cm[0], cm[1]))
            fh.close()   
    print('Scores saved to {}'.format(save_path))

def produce_score_file(dataset, model, device, save_path, batch_size=10):
    data_loader = DataLoader(dataset, batch_size, shuffle=False, drop_last=False)
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    model.is_train = True

    fname_list = []
    key_list = []
    score_list = []
    with torch.no_grad():
        for batch_x, utt_id in tqdm(data_loader):
            fname_list = []
            score_list = []  
            pred_list = []
            batch_size = batch_x.size(0)
            batch_x = batch_x.to(device)
            
            batch_out, batch_emb = model(batch_x)
            score_list.extend(batch_out.data.cpu().numpy().tolist())
            # add outputs
            fname_list.extend(utt_id)

            # score file save into a single file
            with open(os.path.join(save_path, "scores.txt"), 'a+') as fh:
                for f, cm in zip(fname_list,score_list):
                    fh.write('{} {} {}\n'.format(f, cm[0], cm[1]))
            fh.close()   
    print('Scores saved to {}'.format(save_path))
    
def produce_prediction_file(dataset, model, device, save_path):
    data_loader = DataLoader(dataset, batch_size=8, shuffle=False, drop_last=False)
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    
    fname_list = []
    key_list = []
    score_list = []
    with torch.no_grad():
        for batch_x, utt_id in tqdm(data_loader):
            fname_list = []
            score_list = [] 
            pred_list = [] 
            batch_size = batch_x.size(0)
            batch_x = batch_x.to(device)
            
            batch_out = model(batch_x)
            
            batch_score = (batch_out[:, 1]  
                        ).data.cpu().numpy().ravel() 
            batch_prob = nn.Softmax(dim=1)(batch_out)

            # add outputs
            fname_list.extend(utt_id)
            score_list.extend(batch_score.tolist())
            pred_list.extend(batch_prob.tolist())
            
            with open(save_path, 'a+') as fh:
                for f, cm, pred in zip(fname_list,score_list, pred_list):
                    fh.write('{} {} {}\n'.format(f, cm, pred[0]*100))
            fh.close()   
    print('Scores saved to {}'.format(save_path))

def produce_evaluation_file(dataset, model, device, save_path):
    data_loader = DataLoader(dataset, batch_size=10, shuffle=False, drop_last=False)
    num_correct = 0.0
    num_total = 0.0
    model.eval()
    model.is_train = True
    
    fname_list = []
    key_list = []
    score_list = []
    with torch.no_grad():
        for batch_x, utt_id in tqdm(data_loader):
            fname_list = []
            score_list = []  
            batch_size = batch_x.size(0)
            batch_x = batch_x.to(device)
            
            batch_out, _ = model(batch_x)
            
            # batch_score = (batch_out[:, 1]  
            #                ).data.cpu().numpy().ravel() 
            # add outputs
            fname_list.extend(utt_id)
            score_list.extend(batch_out.tolist())
            
            with open(save_path, 'a+') as fh:
                for f, cm in zip(fname_list,score_list):
                    fh.write('{} {} {}\n'.format(f, cm[0], cm[1]))
            fh.close()   
    print('Scores saved to {}'.format(save_path))

def train_epoch(train_loader, model, lr, optim, device):
    running_loss = 0
    num_total = 0.0
    num_correct = 0.0
    model.train()

    #set objective (Loss) functions
    weight = torch.FloatTensor([0.5, 0.5]).to('cuda')
    criterion = nn.CrossEntropyLoss(weight=weight)
    
    for batch_x, batch_y in tqdm(train_loader):
        if torch.cuda.device_count() > 1:
            batch_x = batch_x.to('cuda')
            batch_y = batch_y.view(-1).type(torch.int64).to('cuda')
        else:
            batch_x = batch_x.to(device)
            batch_y = batch_y.view(-1).type(torch.int64).to(device)
       
        batch_size = batch_x.size(0)
        num_total += batch_size
        batch_out, _ = model(batch_x)
        
        batch_loss = criterion(batch_out, batch_y)
        
        running_loss += (batch_loss.item() * batch_size)
        _, batch_pred = batch_out.max(dim=1)
        num_correct += (batch_pred == batch_y).sum(dim=0).item()
        optim.zero_grad()
        batch_loss.backward()
        optim.step()
       
    running_loss /= num_total
    train_accuracy = (num_correct/num_total)*100
    return running_loss, train_accuracy

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ASVspoof2021 baseline system')
    # Dataset
    parser.add_argument('--database_path', type=str, default='/your/path/to/data/', help='Database set')
    '''
    % database_path/
    %   | - protocol.txt
    %   | - audio_path
    
    protocol.txt has 3 columns: key, subset, label
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
    parser.add_argument('--custom', action='store_true', default=False,help='custom eval only')
    parser.add_argument('--eval_part', type=int, default=0)
    parser.add_argument('--predict', action='store_true', default=False,
                        help='get the predicted label instead of score')
    parser.add_argument('--emb', action='store_true', default=False,
                        help='get the embedding instead of score')
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
    

    if not os.path.exists('models'):
        os.mkdir('models')
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
    model_save_path = os.path.join('models', model_tag)

    #set model save directory
    if not os.path.exists(model_save_path):
        os.mkdir(model_save_path)
    
    #GPU device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'                  
    print('Device: {}'.format(device))
    
    model = Model(args,device)
    model = model.to(device)
    nb_params = sum([param.view(-1).size()[0] for param in model.parameters()])
    print('nb_params:',nb_params)

    #set Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr,weight_decay=args.weight_decay)

    if args.model_path:
        try:
            model.load_state_dict(torch.load(args.model_path,map_location=device))
        except:
            print('DataParallel enabled!')
            model = Model(args,device)
            model = model.to(device)
            model = nn.DataParallel(model)
            model.load_state_dict(torch.load(args.model_path,map_location=device))
        
        print('Model loaded : {}'.format(args.model_path))
        
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    #evaluation 
    if args.eval:
        if args.custom:
            file_eval = genSpoof_list_custom(dir_meta = os.path.join(args.database_path+'/protocol.txt'))
        else:
            file_eval = genSpoof_list( dir_meta =  os.path.join(args.database_path+'/protocol.txt'),is_train=False,is_eval=True)
        print('no. of eval trials',len(file_eval))
        eval_set=Dataset_for_eval(list_IDs = file_eval,base_dir = os.path.join(args.database_path+'/'))
        if (args.predict):
            produce_prediction_file(eval_set, model, device, args.eval_output)
        elif (args.emb):
            produce_emb_file(eval_set, model, device, args.eval_output, batch_size=args.batch_size)
        else:
            produce_evaluation_file(eval_set, model, device, args.eval_output)
        sys.exit(0)
   
     
    # define train dataloader
    d_label_trn,file_train = genSpoof_list( dir_meta =  os.path.join(args.database_path,'protocol.txt'),is_train=True,is_eval=False,is_dev=False)
    
    print('no. of training trials',len(file_train))
    
    train_set=Dataset_for(args,list_IDs = file_train,labels = d_label_trn,base_dir = args.database_path+'/',algo=args.algo)
    
    train_loader = DataLoader(train_set, batch_size=args.batch_size,num_workers=8, shuffle=True,drop_last = True)
    
    del train_set,d_label_trn
    

    # define validation dataloader

    d_label_dev,file_dev = genSpoof_list(dir_meta = os.path.join(args.database_path,'protocol.txt'),is_train=False,is_eval=False, is_dev=True)
    
    print('no. of validation trials',len(file_dev))
    
    dev_set = Dataset_for(args,list_IDs = file_dev,
		labels = d_label_dev,
		base_dir = args.database_path+'/',algo=args.algo)
    dev_loader = DataLoader(dev_set, batch_size=args.batch_size,num_workers=8, shuffle=False)
    del dev_set,d_label_dev

    
    

    # Training and validation 
    num_epochs = args.num_epochs
    writer = SummaryWriter('logs/{}'.format(model_tag))
    early_stopping = EarlyStop(patience=10, delta=0.01, init_best=90.0, save_dir=model_save_path)

    for epoch in range(num_epochs):
        
        running_loss, train_accuracy = train_epoch(train_loader,model, args.lr,optimizer, device)
        val_loss, val_accuracy = evaluate_accuracy(dev_loader, model, device)
        writer.add_scalar('train_accuracy', train_accuracy, epoch)
        writer.add_scalar('val_accuracy', val_accuracy, epoch)
        writer.add_scalar('val_loss', val_loss, epoch)
        writer.add_scalar('loss', running_loss, epoch)
        print('\n{} - {} - {} '.format(epoch,running_loss,val_loss))
        early_stopping(val_accuracy, model, epoch)
        if early_stopping.early_stop:
            print("Early stopping activated.")
            break
        
