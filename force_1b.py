import torch
import numpy as np
from torch.utils.data import DataLoader
from datasets_global import PointDataset
import math
 
def frc_1b(invar_1b,vectors_1b):
    re_no,phi,pdr = torch.split(invar_1b,[1,1,1],dim=-1)
    re_m = (1-phi)*re_no

    f_1 = (10*phi)/((1-phi)**2)+(1-phi)**2*(1+1.5*(phi)**(1/2))

    f_2 = 0.11*phi*(1+phi)-0.00456/(1-phi)**4
    f_2 += ((0.169*(1-phi))+0.0644/(1-phi)**4)*re_m**(-0.343)
    f_2 = f_2*re_m
    
    re_t = (2.108*re_m**(0.85)*pdr**(-0.5))
    f_3 = 2.98*phi/(1-phi)**2*re_t

    f = (1-phi)*(f_1+f_2+f_3)
    return f*vectors_1b



if __name__ =="__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    phi = 0.0168
    re = 20
    pdr = 2.56
    invar_1b = torch.as_tensor([[re,phi,pdr]],device=device)
    vectors_1b = torch.as_tensor([[1.,0.,0.]],device=device)

    f1 = frc_1b(invar_1b,vectors_1b)
    print(f1)
    print(f1/(1-phi))
    
