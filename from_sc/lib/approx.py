from torch.autograd import Function 
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap
import numpy as np

import pdb
from lib.precision import fp, pow2

Wistia              = cm.get_cmap('Wistia', 256)
newcolors           = Wistia(np.linspace(0,1,256))
white               = np.array([231/256, 224/256, 224/256, 1])
newcolors[:5, :]   = white
sc_cmp              = ListedColormap(newcolors)  


def get_exp_man(x):
    man, exp = torch.frexp(x.abs())
    man, exp = man*2, exp-1 
    
    return exp.float(), man-1

def get_ch_exp(x):
    exp, man = get_exp_man(x) 
    exp_ch = exp.mean(dim=0) 

    return exp_ch

def rmse(x, y):
    x=x.type(torch.float64)
    y=y.type(torch.float64)

    return ((x-y).abs() / (x.abs()+1e-10) ).norm()

def save_2Dhist(x, y, title="COS_SIM", path = "test.png", bins=100):
    fig = plt.figure(figsize=(7,7))
    fig.set_facecolor('white')

    #H, xedges, yedges = np.histogram2d( x, y, bins=(bins,bins))
    h = plt.hist2d(x, y, bins=bins, cmap=sc_cmp)
    cur_ax = plt.gca()
    fig.colorbar(h[3], ax=cur_ax)
    plt.title(title)
    plt.savefig(path)
    plt.cla()
    plt.close()


class AFC(nn.Linear):
    def __init__(self, in_features, out_features, bias, alp=[0,0,0], SD = None, idx = 0 ):
        super().__init__(in_features, out_features, bias)
        self.flv   , self.blv  , self.glv    = alp
        self.SD = SD
        self.idx= idx
        self.iters = 0
        self.br = self.SD.br
        self.br_cos = self.SD.br_cos
        
    def forward(self, x , iters=None): 
        self.iters = iters if iters is not None else 0
        if self.br      : out   = ALP_br.apply(x, self.weight.T, self)
        elif self.br_cos  : out   = ALP_br_cos.apply(x, self.weight.T, self)
        else            : out   = ALP.apply(x, self.weight.T, self)
        return out

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}, alp={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.SD.alp
        )
 
class ALP(Function): 
    @staticmethod
    def forward(ctx, x, w, L): 
        out             = _ALP_mm(x ,w, L.flv)
        ctx.constant    = (x, w, L) 

        if w.size(1) == 1 : L.SD.update_density(out)
        
        return out.type(torch.float32)

    @staticmethod
    def backward(ctx, grad_output):
        x,  w, L = ctx.constant
        
        err     =  _ALP_mm( grad_output, w.T, L.blv)

        #if (L.SD.density is not None) and (L.iters%600==50) :
        #    if grad_output.size(1) == 1 : L.SD.update_std_idx(grad_output)
        #    
        #    idx = L.SD.std_idx  
        #    std = err[idx].expand( err.size() ) 
        #    sim = L.SD.cos(err[L.SD.mask,:] , std[L.SD.mask,:])
        #        
        #    print( f"[INFO] IDX : {idx} | STDSIG : {L.SD.density[idx].item()}" ) 
        #    save_2Dhist(L.SD.density[L.SD.mask,0].cpu().numpy(), sim.abs().cpu().numpy() ,
        #                    title = f"Layer {L.idx} | Iter {L.iters} | STDSIG {L.SD.density[idx].item():.3f} | HighSIM {(sim.abs()>0.85).sum()/sim.numel():.3f}" ,  
        #                    path=f"{L.SD.workspace}{L.iters}_{L.idx}_cos_sim.png")
            
        w_grad  =  _ALP_mm( x.T, grad_output, L.glv) 

        if  L.glv>=8: w_grad = fp( w_grad, L.glv )
        else        : w_grad = w_grad.type(torch.float16)
   

        return err.type(torch.float32), w_grad.type(torch.float32), None

class ALP_br(Function): 
    @staticmethod
    def forward(ctx, x, w, L): 
        out             = _ALP_mm(x ,w, L.flv)
        ctx.constant    = (x, w, L) 
        
        if L.iters > 450 and w.size(1) == 1: L.SD.update_density(out)
        
        return out.type(torch.float32)

    @staticmethod
    def backward(ctx, grad_output):
        x, w, L = ctx.constant
        if w.size(1) == 1 : L.SD.update_density_grad(grad_output)
        
        gt_err = _ALP_mm (grad_output, w.T, L.blv)
        if (L.SD.density is not None) and (L.SD.reuse_layer == L.idx) and (L.iters > 500) :
            batch = grad_output.size(0) 
            batchwise_err = list()
            for b in range(batch) : 
                temp_err = L.SD.access_err_cache(idx= b)
                if temp_err is None : 
                    temp_err = _ALP_mm( grad_output[b].unsqueeze(0), w.T, L.blv)
                    L.SD.update_err_cache(L.SD.density[b], temp_err)
                else : temp_err = temp_err * L.SD.density_err[b]
                batchwise_err.append(temp_err)
            #print(f"[INFO] ERR CACHE HIT {L.SD.hit} / {L.SD.access}  ({L.SD.hit/L.SD.access * 100 :.2f}%)  |  ZERO {L.SD.zero/L.SD.access * 100 : .2f}%")
            err = torch.cat(batchwise_err, dim=0)            
        else : 
            err =  _ALP_mm( grad_output, w.T, L.blv)            
        
        w_grad  =  _ALP_mm( x.T, grad_output, L.glv) 

        if  L.glv>=8: w_grad = fp( w_grad, L.glv )
        else        : w_grad = w_grad.type(torch.float16)
   

        return err.type(torch.float32), w_grad.type(torch.float32), None


class ALP_br_cos(Function): 
    @staticmethod
    def forward(ctx, x, w, L): 
        out             = _ALP_mm(x ,w, L.flv)
        ctx.constant    = (x, w, L) 
        
        if L.iters > 450 : 
            L.SD.ff_mask[L.idx] = (x>0).cpu()
            if w.size(1) == 1 : L.SD.update_density(out)
        
        return out.type(torch.float32)

    @staticmethod
    def backward(ctx, grad_output):
        x, w, L = ctx.constant
        if w.size(1) == 1 : L.SD.update_density_grad(grad_output)
        
        gt_err = _ALP_mm (grad_output, w.T, L.blv)
        if (L.SD.density is not None) and (L.SD.reuse_layer == L.idx) and (L.iters > 500) :
            batch = grad_output.size(0) 
            batchwise_err = list()
            for b in range(batch) : 
                temp_err = L.SD.access_err_cache_cos(idx= b)
                if temp_err is None : 
                    temp_err = _ALP_mm( grad_output[b].unsqueeze(0), w.T, L.blv)
                    L.SD.update_err_cache_cos(b, temp_err)
                else : temp_err = temp_err * L.SD.density_err[b]
                batchwise_err.append(temp_err)
            print(f"[INFO] ERR CACHE HIT {L.SD.hit} / {L.SD.access}  ({L.SD.hit/L.SD.access * 100 :.2f}%)  |  ZERO {L.SD.zero/L.SD.access * 100 : .2f}%")
            err = torch.cat(batchwise_err, dim=0)            
        else : 
            err =  _ALP_mm( grad_output, w.T, L.blv)            
        
        w_grad  =  _ALP_mm( x.T, grad_output, L.glv) 

        if  L.glv>=8: w_grad = fp( w_grad, L.glv )
        else        : w_grad = w_grad.type(torch.float16)
   

        return err.type(torch.float32), w_grad.type(torch.float32), None

@torch.no_grad()
def _ALP_mm(x, y, lv=-1):
    '''
        === Precision ===                       
        0   : torch.float16                     
        8   : FP8                           8.1  : -SN      
        9   : FP9 (1,5,3)                      
        16  : FP16                          16.1 : -SN       
        17  : CFP16 (1,6,9)                 17.1 : -SN       
        18  : BFP16 (1,8,7)                 
        32  : FP32

        === ALP ===
        1   : ALP l1        1.1 : -SN     1.2 : -SN (1,6,9)
        2   : ALP l2        2.1 : -SN     2.2 : -SN (1,6,9)

        === BLP ===
        3   : BLP l1        1.1 : -SN     

    '''
    precision_convert = [0, 8, 8.1, 9, 16, 16.1, 32, 17, 17.1, 18, 18.1]
 
    if   lv<0                       : return  torch.mm( fp(x)       , fp(y)     )
    elif lv in precision_convert    : return  torch.mm( fp(x,lv)    , fp(y,lv)  )

    # ALP w/o SN
    if lv in [1.2, 2.2]     :  x, y    = fp(x,17.1), fp(y,17.1)
    
    if lv in [1.1,  2.1]    :   x, y    = fp(x,16.1), fp(y,16.1)     
    if lv in [3.1     ]     :  x, y    = fp(x, 8.1), fp(y, 8.1) 
    
    if lv in [1, 2] : x, y = x.type(torch.float16), y.type(torch.float16)
    if lv in [3]    : x, y = fp(x, 8), fp(y, 8) 

    lv = int(lv)
    x_m, x_e = torch.frexp(x.abs()) 
    x_m, x_e = x_m * 2, x_e - 1
    x_alp   = x_m - 1 
    
    x_s     = x.sign()  

    och     = y.size(1)
    batch   = x.size(0) 

    lv1_list    = list()
    lv2_list    = list()
    del x, x_m
    for i in range(och) :
        #y_t = y[:,i].expand(x_alp.size())
        y_t = y[:,i]
        
        # Man, Exp from target och 
        y_m, y_e = torch.frexp(y_t.abs()) 
        y_m, y_e = y_m * 2 , y_e - 1 
        y_alp    = y_m - 1    
        del y_m
        y_alp    =  y_alp.expand(x_alp.size())
        y_e      =  y_e.expand(x_alp.size())

        # Sign 
        y_s =   y_t.sign()
        del y_t
        z_s =   x_s * y_s.expand(x_s.size())
        del y_s
        
        # Exp 
        z_e     =  x_e + y_e
        del y_e
 
        #============ LV1 ================
        lv1_m   =  x_alp + y_alp + 1 + torch.min(x_alp, y_alp) 
        z_m1     =  lv1_m * z_s
        if lv%2==1 : del lv1_m
       
        z_t1 =  torch.ldexp(z_m1, z_e).sum(dim=-1)
        lv1_list.append(z_t1.unsqueeze(1))
        del z_m1, z_t1
        torch.cuda.empty_cache()
        if lv%2==1 : continue 
         
        #============ LV2 ================
        lv2_mask    =  (x_alp + y_alp) >= 1
        lv2_add     =  0.5*(torch.max(x_alp, y_alp)-1)*lv2_mask - 0.5*torch.min(x_alp,y_alp) * (~lv2_mask)
        lv2_m       =  lv1_m + lv2_add
        z_m2        =  lv2_m * z_s
        del lv1_m, lv2_m, lv2_mask, lv2_add
        torch.cuda.empty_cache()

        z_t2 = torch.ldexp(z_m2, z_e).sum(dim=-1)
        lv2_list.append(z_t2.unsqueeze(1))
        if lv==2 : continue 

   
    
    if lv%2==1  : out = torch.cat( lv1_list, dim=-1)
    if lv%2==1  : return out

    out = torch.cat( lv2_list, dim=-1)
    if lv==2 : return out



@torch.no_grad()
def _ALP_mm_add(x, y, lv=-1):

    if lv<0 : return {-1 : torch.mm(x,y)}
    if lv==8 : return {8: torch.mm( fp8(x), fp8(y) )}
    if lv==9 : return {9: torch.mm( fp8_(x), fp8_(y) )}
    if lv==10 : return {10: torch.mm( fp8(x), fp8_(y) )}

    # x = torch.ldexp(x_m, x_e)

    x_m, x_e = torch.frexp(x.abs()) 

    x_m, x_e = x_m * 2, x_e - 1
    x_alp   = x_m - 1 
    x_s     = x.sign()  

    och     = y.size(1)
    batch   = x.size(0) 

    lv1_list    = list()
    lv2_list    = list()
    del x, x_m
    for i in range(och) :
        #y_t = y[:,i].expand(x_alp.size())
        y_t = y[:,i]
        
        # Man, Exp from target och 
        y_m, y_e = torch.frexp(y_t.abs()) 
        y_m, y_e = y_m * 2 , y_e - 1 
        y_alp    = y_m - 1    
        del y_m
        y_alp    =  y_alp.expand(x_alp.size())
        y_e      =  y_e.expand(x_alp.size())

        # Sign 
        y_s =   y_t.sign()
        del y_t
        z_s =   x_s * y_s.expand(x_s.size())
        del y_s
        
        # Exp 
        z_e     =  x_e + y_e
        del y_e
 
        #============ LV1 ================
        lv1_m   =  x_alp + y_alp + 1 + torch.min(x_alp, y_alp) 
        z_m1     =  lv1_m * z_s
        if lv==1 : del lv1_m
       
        z_t1 =  torch.ldexp(z_m1, z_e).sum(dim=-1)
        lv1_list.append(z_t1.unsqueeze(1))
        del z_m1, z_t1
        torch.cuda.empty_cache()
        if lv==1 : continue 
         
        #============ LV2 ================
        lv2_mask    =  (x_alp + y_alp) >= 1
        lv2_add     =  0.5*(torch.max(x_alp, y_alp)-1)*lv2_mask - 0.5*torch.min(x_alp,y_alp) * (~lv2_mask)
        lv2_m       =  lv1_m + lv2_add
        z_m2        =  lv2_m * z_s
        del lv1_m, lv2_m, lv2_mask, lv2_add
        torch.cuda.empty_cache()

        z_t2 = torch.ldexp(z_m2, z_e).sum(dim=-1)
        lv2_list.append(z_t2.unsqueeze(1))
        if lv==2 : continue 

   
    out = dict()
    out[1] = torch.cat( lv1_list, dim=-1)
    if lv==1 : return out

    out[2] = torch.cat( lv2_list, dim=-1)
    if lv==2 : return out



