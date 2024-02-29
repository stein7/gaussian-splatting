import torch
from lib.approx import get_exp_man
import numpy as np

import pdb

def d2b_fp16( x ) :
    
    size = x.size()
    sign = x.sign()
    sign_bit = -1*(sign/2+0.5) + 1
    exp, man = get_exp_man(x.abs())
    exp += 14 # (-1)+15

    if exp.min() < 0 or exp.max() > 31 : print( "[WARN] OUT OF FP16 BOUND" )

    man = man*2 -1 
    man = man * (2**10)

    exp_bit = tensor_d2b(exp, 5)
    man_bit = tensor_d2b(man, 10)
    result = np.zeros(( size[0], size[1])).astype(np.string_)
    for i in range(size[0] ) :
        for j in range(size[1]) :
            tar     = str( int(sign_bit[i][j].item()) ) + str(exp_bit[i][j])[2:-1] + str(man_bit[i][j])[2:-1]
            result[i][j] = tar
    return result
    

def tensor_d2b(x, width=10) : 
   
    size= x.size()
    x=x.abs().detach().cpu().numpy().astype(np.int32)
    vbin = np.vectorize(bin)
    bin_array = vbin(x) 

    result = [[0] * size[1] ] * size[0]
    result = np.zeros(( size[0], size[1])).astype(np.string_)
    for i in range(size[0] ) :
        for j in range(size[1]) : 
            tar     = str(bin_array[i][j]).replace("0b","").zfill(width)
            result[i][j]    =   tar
    return result



    
