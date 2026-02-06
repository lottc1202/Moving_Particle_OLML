import torch
#from torch_cluster import radius, radius_graph
#from torch_scatter import scatter
from e3nn import o3,nn
import numpy as np
from torch.utils.data import DataLoader
from datasets_global import PointDataset
import copy
import math
 
class self_intrc(torch.nn.Module):
    def __init__(self,irreps_in,irreps_out,act=torch.tanh):
        super().__init__()
        self.lin = o3.Linear(irreps_in,irreps_out)
        self.non_lin = nn.NormActivation(irreps_out,act,epsilon=1e-6,bias=True)
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out

    def forward(self,x):
        x = self.lin(x)
        x = self.non_lin(x)
        return x

class frc_2b(torch.nn.Module):
    def __init__(self,lyr_wdt=10,num_lyrs=2,
                 act1=torch.nn.ReLU(),
                 act2=torch.tanh):
        super().__init__()
        num_ir = lyr_wdt
        self.num_ir = num_ir
        self.irreps_in = o3.Irreps("3x1o")
        self.irreps_out = o3.Irreps("1x1o")
        self.n_scalars = o3.Irreps(str(num_ir)+"x1o")
        self.act1 = act1
        self.act2 = act2

        # This is for Re,phi,PDR
        lyr_list1 = torch.nn.ModuleList([torch.nn.Linear(5,num_ir)])
        # This is e3nn version for relative position and unit mean flow vector
        lyr_list2 = torch.nn.ModuleList([self_intrc(self.irreps_in,self.n_scalars,
                                           act=act2)])

        for i in range(1,num_lyrs):
            lyr_list1.append(torch.nn.Linear(num_ir,num_ir))
            lyr_list2.append(self_intrc(lyr_list2[i-1].irreps_out,self.n_scalars,
                                          act=act2))

        self.lyr_list1 = lyr_list1
        self.lyr_list2 = lyr_list2

        # Concatenation between branchnet and trunknet is done
        # by the below layer
        self.concat = o3.ElementwiseTensorProduct(str(num_ir)+"x0e",
                                                  str(num_ir)+"x1o")

        self.final_lin = o3.Linear(self.n_scalars,self.irreps_out)

        # This is new as of 9/27/24 and requires the last layer of the NN to have weights set to 1 
       # with torch.no_grad():
       #     self.final_lin.weight.fill_(1)
       #     if self.final_lin.bias is not None: 
       #         self.final_lin.bias.fill_(0)
       # self.final_lin.weight.requires_grad = False 
    
        # Forward function for the NN
    def forward(self,invar_2b,vectors_2b):
        # Change the shape of invar_2b
        invar_2b = invar_2b.reshape(-1,invar_2b.size(dim=-1))

        for i,lyr in enumerate(self.lyr_list1):
            if i == 0:
                y = lyr(invar_2b)
            else:
                y = lyr(y)

            y = self.act1(y)

        y = y.reshape(-1,vectors_2b.size(dim=1),self.num_ir)


        for i,lyr in enumerate(self.lyr_list2):
            if i == 0:
                x = lyr(vectors_2b)
            else:
                x = lyr(x)

        x = self.concat(y,x) 

        x = self.final_lin(x)

        #x = torch.linalg.norm(x, dim=2)
        return  torch.sum(x,dim=(1))

if __name__ =="__main__":
    num_neig = 4
    dataloader = DataLoader(
      PointDataset([range(1,250),],fl_lst=[7,],
                   num_neig_2b=num_neig,
                   num_neig_3b=2),
      batch_size=2,
      shuffle=True,)
 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    mdl = frc_2b(lyr_wdt=50,num_lyrs=6)
    print("The model contains {} parameters".format(sum(p.numel() for p in mdl.parameters() if p.requires_grad)))
    mdl.to(device)

    if torch.cuda.device_count() >1:
        print("Using",torch.cuda.device_count(),"GPUs!")
        mdl = torch.nn.DataParallel(mdl)
        mdl.to(device)

    for i,datas in enumerate(dataloader):
        invar_2b = datas['invar_2b'].to(device)
        vectors_2b = datas['vectors_2b'].to(device)
        f1 = mdl(invar_2b,vectors_2b)
        print(f1) 

        if i == 0:
            break
