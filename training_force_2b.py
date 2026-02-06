import argparse
import os
import numpy as np
import math
import sys
import torch
from torch.utils.data import DataLoader
import scipy.io as sio
from sklearn.metrics import r2_score

from force_1b import *
from force_2b import *
from datasets_global import PointDataset

#import utils # This requires visdom 

# Change the default values in the parser according to requirements 
parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--num_neig",type=int,default=26,help="Number of sorted neighbors to consider")
parser.add_argument("--num_lyrs",type=int,default=6,help="number of layer in model")
parser.add_argument("--lyr_wdt",type=int,default=100,help="width of each layer")
parser.add_argument("--k_fold",type=int,default=5,help="K-fold cross-validation")
parser.add_argument("--num_fold",type=int,default=1,help="Fold of consideration")
parser.add_argument("--start_fold",type=int,default=0,help="Fold of consideration")
parser.add_argument("--frac_train",type=float,default=1.0,help="percentage of training data to use")
parser.add_argument("--prt",type=int,default=8097,help="Port number for visdom")  
parser.add_argument("--batch_size", type=int, default=100, help="size of the batches")
parser.add_argument("--lr", type=float, default=1e-3, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient squares")
parser.add_argument("--n_cpu", type=int, default=4, help="number of cpu threads to use during batch generation")
opt = parser.parse_args()
print(opt)

fl_lst = list(range(1,49))

os.makedirs("saved_models/", exist_ok=True)
os.makedirs("loss_evolution/", exist_ok=True)

# Deciding to use GPUs or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Giving higher weightage to lift directions (y,z)
w_lft = torch.as_tensor([[1.0,1.0,1.0]],dtype=torch.float,device=device) 

def saving_models(eph,fold_ind,mdl,optmzr,schdlr,mdl_type='_force_2b'):
    save_path = "saved_models/"+"bx_sz_LN_26"
    save_path = save_path + "_neig_"+str(opt.num_neig)
    save_path = save_path + "_kfold_"+str(opt.k_fold)+"_"+str(fold_ind)
    save_path = save_path + "_l_"+str(opt.num_lyrs)+"_w_"+str(opt.lyr_wdt)
    save_path = save_path+mdl_type+".tar"
    if torch.cuda.device_count() >1:
        torch.save({'epoch':eph,
                'mdl':mdl.module.state_dict(),
                'optmzr':optmzr.state_dict(),
                'schdlr':schdlr.state_dict(), 
               },save_path)
    else:
        torch.save({'epoch':eph,
                'mdl':mdl.state_dict(),
                'optmzr':optmzr.state_dict(),
                'schdlr':schdlr.state_dict(), 
               },save_path)

#global plotter
#plotter = utils.VisdomLinePlotter(env_name='force',prt=str(opt.prt))

# Loss function
criterion_loss = torch.nn.MSELoss()

# Cross-validation loop 
for fld_i in range(opt.num_fold):
    l_t = []
    l_v = []

    mdl_2b = frc_2b(lyr_wdt=opt.lyr_wdt,
                    num_lyrs=opt.num_lyrs,
                    act1=torch.nn.ReLU(),
                    act2=torch.tanh)

    # Having optimizers and scheduler on CPU
    # Optimizer

    optimizer_2b = torch.optim.Adam(mdl_2b.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
    scheduler_2b = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_2b,patience=10,factor=0.1)
    mdl_2b.to(device)

    if torch.cuda.device_count() >1:
        print("Using",torch.cuda.device_count(),"GPUs!")
        mdl_2b = torch.nn.DataParallel(mdl_2b)
        mdl_2b.to(device)

    global_train_lst = []
    global_val_lst = []

    for fl_no in fl_lst:
        # Total number of samples in each dataset
        fl_nm = "../data/"+"GCN_Input_"+str(fl_no)+".txt"
        data_matrix = np.loadtxt(fl_nm)
        dt_list = range(np.size(data_matrix,0))
        # Figuring out test samples
        test_sz = int(len(dt_list)/opt.k_fold)
        test_lst = range(opt.start_fold*test_sz,(opt.start_fold+1)*test_sz)
    
        train_val_lst = [x for x in dt_list if x not in test_lst]
    
        # Divide this into 80-20
        train_lst = train_val_lst[:int(0.8*opt.frac_train*len(train_val_lst))]
        val_lst = train_val_lst[int(0.8*len(train_val_lst)):]

        global_train_lst.append(train_lst)
        global_val_lst.append(val_lst)

    # Creating DataLoader for training
    dataloader = DataLoader(
      PointDataset(global_train_lst,fl_lst=fl_lst,
       num_neig_2b=opt.num_neig,
       num_neig_3b=2),
      batch_size=opt.batch_size,
      shuffle=True,
      num_workers=opt.n_cpu)

    # Creating Dataloader for validation
    val_dataloader = DataLoader(
      PointDataset(global_val_lst,fl_lst=fl_lst,
       num_neig_2b=opt.num_neig,
       num_neig_3b=2),
      batch_size=opt.batch_size,
      shuffle=True,
      num_workers=opt.n_cpu)
    # -----------------------
    # Training 2-Body
    # -----------------------

    val_cnt = 0 # Checker to stop based on validation set 
    prev_val = 10 # A random number to get started for validation loss
    for epoch in range(opt.n_epochs):
        mean_scheduler = 0.0 # This metric is used for Learning rate scheduler
        cur_val = 0.0 # A variable to figure out current validation loss
        mdl_2b.train(mode=True)
        for i,datas in enumerate(dataloader):
            # Model inputs
            invar_1b = datas['invar_1b'].to(device)
            vectors_1b = datas['vectors_1b'].to(device)

            invar_2b = datas['invar_2b'].to(device)
            vectors_2b = datas['vectors_2b'].to(device)

            mean_drag = datas['mu_d'].to(device)
            std_drag  = datas['std_d'].to(device)
          
            # ---------------------------
            # Training
            # ---------------------------
            
            fake = mdl_2b(invar_2b,vectors_2b)
            #fake = fake.unsqueeze(1)

            real = datas['frc'].to(device)
            #real = real - (frc_1b(invar_1b,vectors_1b))
            real  = (real - frc_1b(invar_1b,vectors_1b))

            #wgt = datas['wgt_f'].to(device)
            wgt = 1/(mean_drag)

            # Loss
            loss_2b = criterion_loss(wgt*fake, wgt*real)
            optimizer_2b.zero_grad()  # Clearing out previous gradients
            loss_2b.backward()
            optimizer_2b.step()

            batches_done = epoch * len(dataloader) + i

            # Learning Rate scheduling metric
            mean_scheduler = mean_scheduler + loss_2b.item()

        mean_scheduler = mean_scheduler/len(dataloader)
        real_np        = real.detach().cpu().numpy()
        fake_np        = fake.detach().cpu().numpy()
        train_acc      = r2_score(real_np[:,1],fake_np[:,1])

        # Validation loss
        mdl_2b.train(mode=False)
        with torch.no_grad():
            for i,datas in enumerate(val_dataloader):
                invar_1b = datas['invar_1b'].to(device)
                vectors_1b = datas['vectors_1b'].to(device)

                invar_2b = datas['invar_2b'].to(device)
                vectors_2b = datas['vectors_2b'].to(device)
                
                mean_drag = datas['mu_d'].to(device)
                std_drag  = datas['std_d'].to(device)
                
                fake = mdl_2b(invar_2b,vectors_2b)
                #fake = fake.unsqueeze(1)
                real = datas['frc'].to(device)
                #real = real - (frc_1b(invar_1b,vectors_1b))

                real  = (real - frc_1b(invar_1b,vectors_1b))
                #wgt = datas['wgt_f'].to(device)
                wgt = 1/(mean_drag)
                
                loss_val = criterion_loss(wgt*fake, wgt*real)
                cur_val = cur_val + loss_val
    
        cur_val = cur_val/len(val_dataloader)
        real_np = real.detach().cpu().numpy()
        fake_np = fake.detach().cpu().numpy()
        val_acc = r2_score(real_np[:,1],fake_np[:,1])

        print(f"Epoch: {epoch+1}/{opt.n_epochs}, Training Loss: {mean_scheduler:.3f}, Training R^2: {train_acc:.4f},  Validation Loss    : {cur_val:.3f}, Validation R^2: {val_acc:.4f}\n",flush=True)

        # Store these values in lists
        l_t.append(mean_scheduler)
        l_v.append(cur_val.item())

        # Scheduler for Generator
        """
        if (epoch % 20 == 0):
            plotter.plot('2b_loss_epoch'+str(fld_i),'Training','TwoBody',epoch+1,mean_scheduler)
            plotter.plot('2b_loss_epoch'+str(fld_i),'Validation','TwoBody',epoch+1,cur_val.item())
        """
        scheduler_2b.step(mean_scheduler) # Scheduler to reduce learning rate

        if epoch == 0:
            prev_val = cur_val + 10

        if (cur_val>=prev_val) and (prev_val > mean_scheduler):
            val_cnt = val_cnt + 1
            if (val_cnt >= 50) and (epoch > 99):
                saving_models(epoch+1,opt.start_fold,mdl_2b,optimizer_2b,
                              scheduler_2b,mdl_type='_force_2b_Tang_Surrogate')
                break
        elif (cur_val < prev_val):
            prev_val = cur_val
            val_cnt = 0
        else:
            val_cnt = 0

        if epoch == opt.n_epochs-1:
            saving_models(epoch+1,opt.start_fold,mdl_2b,optimizer_2b,
                              scheduler_2b,mdl_type='_force_2b_Tang_Surrogate')

    # Convert the loss lists into numpy
    l_t = np.array(l_t)
    l_v = np.array(l_v)
    l_t = np.float32(l_t)
    l_v = np.float32(l_v)
    thisdict = {'train_l':l_t,'val_l':l_v}
    sio.savemat("loss_evolution/bx_sz_LN_26_neig_"+
                 str(opt.num_neig)+"_kfold_"+
                 str(opt.k_fold)+"_"+str(opt.start_fold)+
                 "_l_"+str(opt.num_lyrs)+"_w_"+
                 str(opt.lyr_wdt)+"_2b.mat",thisdict)
