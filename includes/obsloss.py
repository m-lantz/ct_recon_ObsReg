import numpy as np
import torch

class ObsLoss(torch.nn.Module):
    
    def __init__(self,lam,coef,Uf,learn_filter,patch_size,lossfcn,device):
        super(ObsLoss,self).__init__()
        self.lam = torch.tensor(lam).float().to(device)
        self.coef = torch.nn.Parameter(torch.from_numpy(coef).float().to(device), requires_grad = learn_filter)
        self.Uf = torch.tensor(Uf).float().to(device)
        self.patch_size = patch_size
        self.lossfcn = lossfcn
        
    def forward(self,img_recon_sig,img_recon_nosig,img_clean_sig,img_clean_nosig,img_bg_sig,img_bg_nosig,mask_sig,mask_nosig):
        ps = self.patch_size

        img_small_sig = torch.reshape(img_recon_sig[mask_sig]-img_bg_sig[mask_sig],(-1,ps[0],ps[1]))
        img_small_nosig = torch.reshape(img_recon_nosig[mask_nosig]-img_bg_nosig[mask_nosig],(-1,ps[0],ps[1]))

        # Compte MSE Loss
        loss1 = self.lossfcn(torch.cat([img_recon_sig,img_recon_nosig]),torch.cat([img_clean_sig,img_clean_nosig]))

        # Compute observer Loss
        g1bar = torch.mean(img_small_sig,0)
        g0bar = torch.mean(img_small_nosig,0)
        gdelta = torch.flatten(g1bar-g0bar)
        w = torch.matmul(self.coef,self.Uf) #template


        num = torch.square(torch.dot(w,gdelta).view(-1,))
        wK1w = torch.mean(torch.square(torch.matmul(w,torch.reshape(img_small_sig-g1bar, (np.prod(ps),-1))))).view(-1,)
        wK0w = torch.mean(torch.square(torch.matmul(w,torch.reshape(img_small_nosig-g0bar, (np.prod(ps),-1))))).view(-1,)
        denom = 0.5*(wK1w + wK0w)
        loss2 = torch.div(num,denom) #SNR squared
        # loss2 = torch.sqrt(torch.div(num,denom)) #SNR
        loss = loss1-self.lam*loss2
        return loss,loss1,loss2

class ObsLossMatchedSig(torch.nn.Module):
    
    def __init__(self,lam,patch_size,lossfcn,device):
        super(ObsLossMatchedSig,self).__init__()
        self.lam = torch.tensor(lam).float().to(device)
        self.patch_size = patch_size
        self.lossfcn = lossfcn
        
    def forward(self,img_recon_sig,img_recon_nosig,img_clean_sig,img_clean_nosig,img_bg_sig,img_bg_nosig,img_sigonly_sig,img_sigonly_nosig,mask_sig,mask_nosig):
        ps = self.patch_size

        img_small_sig = torch.reshape(img_recon_sig[mask_sig]-img_bg_sig[mask_sig],(-1,ps[0],ps[1]))
        img_small_nosig = torch.reshape(img_recon_nosig[mask_nosig]-img_bg_nosig[mask_nosig],(-1,ps[0],ps[1]))
        img_small_sigonly_sig = torch.reshape(img_sigonly_sig[mask_sig],(-1,ps[0],ps[1]))
        img_small_sigonly_nosig = torch.reshape(img_sigonly_nosig[mask_nosig],(-1,ps[0],ps[1]))

        # Compte MSE Loss
        loss1 = self.lossfcn(torch.cat([img_recon_sig,img_recon_nosig]),torch.cat([img_clean_sig,img_clean_nosig]))

        # Compute observer Loss
        # g1bar = torch.mean(img_small_sig,0)
        # g0bar = torch.mean(img_small_nosig,0)
        # gdelta = torch.flatten(g1bar-g0bar)
        # w1 = torch.flatten(torch.mean(img_small_sigonly_sig,0))
        # w0 = torch.flatten(torch.mean(img_small_sigonly_nosig,0))
        # w = (w0+w1)/2.0
        g1bar = torch.dot(torch.flatten(img_small_sig),torch.flatten(img_small_sigonly_sig))
        g0bar = torch.dot(torch.flatten(img_small_nosig),torch.flatten(img_small_sigonly_nosig))
        loss2 = g1bar-g0bar
        # wK1w = torch.mean(torch.square(torch))

        # num = torch.square(torch.dot(w,gdelta).view(-1,))
        # wK1w = torch.mean(torch.square(torch.matmul(w1,torch.reshape(img_small_sig-g1bar, (np.prod(ps),-1))))).view(-1,)
        # wK0w = torch.mean(torch.square(torch.matmul(w0,torch.reshape(img_small_nosig-g0bar, (np.prod(ps),-1))))).view(-1,)
        # denom = 0.5*(wK1w + wK0w)
        # loss2 = torch.div(num,denom) #SNR squared
        # loss2 = torch.sqrt(torch.div(num,denom)) #SNR
        loss = loss1-self.lam*loss2
        # return loss,loss1,loss2,img_small_sigonly_sig,img_small_sigonly_nosig
        return loss,loss1,loss2

class ObsLossRefactor(torch.nn.Module):
    def __init__(self,img_size,patch_size,lossfcn,lam,device):
        super(ObsLossRefactor,self).__init__()
        self.device = device
        self.lam = torch.tensor(lam).float().to(device) 
        self.img_size = img_size
        self.patch_size = patch_size
        self.lossfcn = lossfcn
        
    def forward(self,data,net,lossfcn):
        device = self.device

        img_noisy_sig = data["sig_real"].to(device)
        img_clean_sig = data["sig_image"].to(device)
        img_bg_sig    = data["nosig_image"].to(device)
        img_sigonly_sig = data["sig_only"].to(device)
        img_coordinates_sig = data["sig_coordinates"].to(device)

        img_noisy_nosig = data["nosig_real"].to(device)
        img_clean_nosig = data["nosig_image"].to(device)
        img_bg_nosig    = data["nosig_image"].to(device)
        img_sigonly_nosig = data["sig_only"].to(device)
        img_coordinates_nosig = data["sig_coordinates"].to(device)

        img_recon_sig = net(img_noisy_sig)
        img_recon_nosig = net(img_noisy_nosig)

        mask_sig = torch.zeros_like(img_noisy_sig).bool()
        mask_nosig = torch.zeros_like(img_noisy_sig).bool()

        isize = self.img_size
        ix = int(isize[0]/2)
        iy = int(isize[1]/2)

        psize = self.patch_size
        px = int(psize[0]/2)
        py = int(psize[1]/2)
        
        for k in range(len(img_noisy_sig)):
            xx,yy = img_coordinates_sig[k,:]
            mask_sig[k,:,(xx-px):(xx+px),(yy-py):(yy+py)] = True 
            xx,yy = img_coordinates_nosig[k,:]
            mask_nosig[k,:,(xx-px):(xx+px),(yy-py):(yy+py)] = True

        img_small_sig = torch.reshape(img_recon_sig[mask_sig]-img_bg_sig[mask_sig],(-1,psize[0],psize[1]))
        img_small_nosig = torch.reshape(img_recon_nosig[mask_nosig]-img_bg_nosig[mask_nosig],(-1,psize[0],psize[1]))
        img_small_sigonly_sig = torch.reshape(img_sigonly_sig[mask_sig],(-1,psize[0],psize[1]))
        img_small_sigonly_nosig = torch.reshape(img_sigonly_nosig[mask_nosig],(-1,psize[0],psize[1]))

        # Compte MSE Loss
        loss1 = self.lossfcn(torch.cat([img_recon_sig,img_recon_nosig]),torch.cat([img_clean_sig,img_clean_nosig]))

        # Compute observer Loss
        # g1bar = torch.mean(img_small_sig,0)
        # g0bar = torch.mean(img_small_nosig,0)
        # gdelta = torch.flatten(g1bar-g0bar)
        # w1 = torch.flatten(torch.mean(img_small_sigonly_sig,0))
        # w0 = torch.flatten(torch.mean(img_small_sigonly_nosig,0))
        # w = (w0+w1)/2.0
        g1bar = torch.dot(torch.flatten(img_small_sig),torch.flatten(img_small_sigonly_sig))
        g0bar = torch.dot(torch.flatten(img_small_nosig),torch.flatten(img_small_sigonly_nosig)) #this doesn't make sense...
        # sig_only_no_sig is an empty array...?
        loss2 = g1bar-g0bar
        # wK1w = torch.mean(torch.square(torch))

        # num = torch.square(torch.dot(w,gdelta).view(-1,))
        # wK1w = torch.mean(torch.square(torch.matmul(w1,torch.reshape(img_small_sig-g1bar, (np.prod(ps),-1))))).view(-1,)
        # wK0w = torch.mean(torch.square(torch.matmul(w0,torch.reshape(img_small_nosig-g0bar, (np.prod(ps),-1))))).view(-1,)
        # denom = 0.5*(wK1w + wK0w)
        # loss2 = torch.div(num,denom) #SNR squared
        # loss2 = torch.sqrt(torch.div(num,denom)) #SNR
        loss = loss1-self.lam*loss2
        # return loss,loss1,loss2,img_small_sigonly_sig,img_small_sigonly_nosig
        return loss,loss1,loss2

class ObsLossRefactorNew(torch.nn.Module):
    def __init__(self,img_size,patch_size,lossfcn,lam,device):
        super(ObsLossRefactorNew,self).__init__()
        self.device = device
        self.lam = torch.tensor(lam).float().to(device) 
        self.img_size = img_size
        self.patch_size = patch_size
        self.lossfcn = lossfcn
        
    def forward(self,data,net,lossfcn):
        device = self.device

        img_noisy_sig = data["sig_real"].to(device)
        img_noisy_nosig = data["nosig_real"].to(device)
        img_clean_nosig = data["nosig_image"].to(device)
        img_sigonly_sig = data["sig_only"].to(device)
        img_sigonly = data["sig_only"].to(device)
        img_coordinates_sig = data["sig_coordinates"].to(device)

        img_recon_sig = net(img_noisy_sig)
        img_recon_nosig = net(img_noisy_nosig)

        mask_sig = torch.zeros_like(img_noisy_sig).bool()
    
        isize = self.img_size
        ix = int(isize[0]/2)
        iy = int(isize[1]/2)

        psize = self.patch_size
        px = int(psize[0]/2)
        py = int(psize[1]/2)
        
        for k in range(len(img_noisy_sig)):
            xx,yy = img_coordinates_sig[k,:]
            mask_sig[k,:,(xx-px):(xx+px),(yy-py):(yy+py)] = True 
 
        img_small_sig = torch.reshape(img_recon_sig[mask_sig],(-1,psize[0],psize[1]))
        img_small_nosig = torch.reshape(img_recon_nosig[mask_sig],(-1,psize[0],psize[1]))
        img_small_sigonly = torch.reshape(img_sigonly[mask_sig],(-1,psize[0],psize[1]))
 
        # Compute loss using passed in lossfcn (with respect to nosig images only)
        loss1 = self.lossfcn(img_recon_nosig,img_clean_nosig)

        # Compute observer Loss
        g1bar = torch.dot(torch.flatten(img_small_sig),torch.flatten(img_small_sigonly))
        g0bar = torch.dot(torch.flatten(img_small_nosig),torch.flatten(img_small_sigonly)) #this doesn't make sense...
        loss2 = g1bar-g0bar

        loss = loss1-self.lam*loss2

        return loss,loss1,loss2

class ObsLossRefactorSigInd(torch.nn.Module):
    def __init__(self,img_size,patch_size,lossfcn,lam,device):
        super(ObsLossRefactorSigInd,self).__init__()
        self.device = device
        self.lam = torch.tensor(lam).float().to(device) 
        self.img_size = img_size
        self.patch_size = patch_size
        self.lossfcn = lossfcn
        
    def forward(self,data,net,lossfcn):
        device = self.device

        img_noisy_nosig = data[0]["nosig_real"].to(device)
        img_clean_nosig = data[0]["nosig_image"].to(device)
        img_sigonly = data[1]["sig_template"].to(device)
        img_sigfbp = data[1]["sig_fbp"].to(device)
        img_coordinates_sig = data[1]["sig_coordinates"].to(device)
        img_noisy_sig = img_noisy_nosig+img_sigfbp

        img_recon_sig = net(img_noisy_sig)
        img_recon_nosig = net(img_noisy_nosig)

        mask_sig = torch.zeros_like(img_noisy_sig).bool()
    
        isize = self.img_size
        ix = int(isize[0]/2)
        iy = int(isize[1]/2)

        psize = self.patch_size
        px = int(psize[0]/2)
        py = int(psize[1]/2)
        
        for k in range(len(img_noisy_sig)):
            xx,yy = img_coordinates_sig[k,:]
            mask_sig[k,:,(xx-px):(xx+px),(yy-py):(yy+py)] = True 
 
        img_small_sig = torch.reshape(img_recon_sig[mask_sig],(-1,psize[0],psize[1]))
        img_small_nosig = torch.reshape(img_recon_nosig[mask_sig],(-1,psize[0],psize[1]))
        img_small_sigonly = torch.reshape(img_sigonly[mask_sig],(-1,psize[0],psize[1]))
 
        # Compute loss using passed in lossfcn (with respect to nosig images only)
        loss1 = self.lossfcn(img_recon_nosig,img_clean_nosig)

        # Compute observer Loss
        g1bar = torch.dot(torch.flatten(img_small_sig),torch.flatten(img_small_sigonly))
        g0bar = torch.dot(torch.flatten(img_small_nosig),torch.flatten(img_small_sigonly)) #this doesn't make sense...
        loss2 = g1bar-g0bar

        loss = loss1-self.lam*loss2

        return loss,loss1,loss2

class ObsLossRefactorCHO(torch.nn.Module):
    def __init__(self,coef,Uf,learn_filter,img_size,patch_size,lossfcn,lam,device):
        super(ObsLossRefactorCHO,self).__init__()
        self.lam = torch.tensor(lam).float().to(device)
        self.coef = torch.nn.Parameter(torch.from_numpy(coef).float().to(device), requires_grad = learn_filter) #CHO coefficients
        self.Uf = torch.tensor(Uf).float().to(device) #CHO channels
        self.patch_size = patch_size
        self.img_size = img_size
        self.lossfcn = lossfcn
        self.device = device
          
    def forward(self,data,net,lossfcn):
        device = self.device

        img_noisy_sig = data["sig_real"].to(device)
        img_clean_sig = data["sig_image"].to(device)
        img_bg_sig    = data["nosig_image"].to(device)
        img_coordinates_sig = data["sig_coordinates"].to(device)

        img_noisy_nosig = data["nosig_real"].to(device)
        img_clean_nosig = data["nosig_image"].to(device)
        img_bg_nosig    = data["nosig_image"].to(device)

        img_recon_sig = net(img_noisy_sig)
        img_recon_nosig = net(img_noisy_nosig)

        mask_sig = torch.zeros_like(img_noisy_sig).bool()

        isize = self.img_size
        ix = int(isize[0]/2)
        iy = int(isize[1]/2)

        psize = self.patch_size
        px = int(psize[0]/2)
        py = int(psize[1]/2)
        
        for k in range(len(img_noisy_sig)):
            xx,yy = img_coordinates_sig[k,:]
            mask_sig[k,:,(xx-px):(xx+px),(yy-py):(yy+py)] = True 

        img_small_sig = torch.reshape(img_recon_sig[mask_sig]-img_bg_sig[mask_sig],(-1,psize[0],psize[1]))
        img_small_nosig = torch.reshape(img_recon_nosig[mask_sig]-img_bg_nosig[mask_sig],(-1,psize[0],psize[1]))
   
        # Compte MSE loss
        loss1 = self.lossfcn(img_recon_nosig,img_clean_nosig)

        # Compute observer loss       
        g1bar = torch.mean(img_small_sig,0)
        g0bar = torch.mean(img_small_nosig,0)
        gdelta = torch.flatten(g1bar-g0bar)
        w = torch.matmul(self.coef,self.Uf) #template

        num = torch.square(torch.dot(w,gdelta).view(-1,))
        wK1w = torch.mean(torch.square(torch.matmul(w,torch.reshape(img_small_sig-g1bar, (np.prod(psize),-1))))).view(-1,)
        wK0w = torch.mean(torch.square(torch.matmul(w,torch.reshape(img_small_nosig-g0bar, (np.prod(psize),-1))))).view(-1,)
        denom = 0.5*(wK1w + wK0w)
        # loss2 = torch.div(num,denom) #SNR squared
        loss2 = torch.sqrt(torch.div(num,denom)) #SNR
        loss = loss1-self.lam*loss2
        return loss,loss1,loss2

class ObsLossRefactorLinear(torch.nn.Module):
    def __init__(self,w,img_size,r,lossfcn,lam,device):
        super(ObsLossRefactorLinear,self).__init__()
        self.lam = torch.tensor(lam).float().to(device)
        self.w = torch.tensor(w).float().to(device) #template
        self.r = r
        self.img_size = img_size
        self.lossfcn = lossfcn
        self.device = device
          
    def forward(self,data,net,lossfcn):
        device = self.device

        img_noisy_sig = data["sig_real"].to(device)
        img_clean_sig = data["sig_image"].to(device)
        img_bg_sig    = data["nosig_image"].to(device)
        img_coordinates_sig = data["sig_coordinates"].to(device)

        img_noisy_nosig = data["nosig_real"].to(device)
        img_clean_nosig = data["nosig_image"].to(device)
        img_bg_nosig    = data["nosig_image"].to(device)

        img_recon_sig = net(img_noisy_sig)
        img_recon_nosig = net(img_noisy_nosig)

        mask_sig = torch.zeros_like(img_noisy_sig).bool()

        isize = self.img_size
        ix = int(isize[0]/2)
        iy = int(isize[1]/2)

        # psize = self.patch_size
        # px = int(psize[0]/2)
        # py = int(psize[1]/2)
        r = self.r
        
        for k in range(len(img_noisy_sig)):
            xx,yy = img_coordinates_sig[k,:]
            mask_sig[k,:,(xx-px):(xx+px),(yy-py):(yy+py)] = True 

        img_small_sig = torch.reshape(img_recon_sig[mask_sig]-img_bg_sig[mask_sig],(-1,2*r-1,2*r-1))
        img_small_nosig = torch.reshape(img_recon_nosig[mask_sig]-img_bg_nosig[mask_sig],(-1,2*r-1,2*r-1))
   
        # Compte MSE loss
        loss1 = self.lossfcn(img_recon_nosig,img_clean_nosig)

        # Compute observer loss       
        g1bar = torch.mean(img_small_sig,0)
        g0bar = torch.mean(img_small_nosig,0)
        gdelta = torch.flatten(g1bar-g0bar)
        w = torch.flatten(self.w)

        num = torch.square(torch.dot(w,gdelta).view(-1,))
        wK1w = torch.mean(torch.square(torch.matmul(w,torch.reshape(img_small_sig-g1bar, ((2*r-1)*(2*r-1),-1))))).view(-1,)
        wK0w = torch.mean(torch.square(torch.matmul(w,torch.reshape(img_small_nosig-g0bar, ((2*r-1)*(2*r-1),-1))))).view(-1,)
        denom = 0.5*(wK1w + wK0w)
        # loss2 = torch.div(num,denom) #SNR squared
        loss2 = torch.sqrt(torch.div(num,denom)) #SNR
        loss = loss1-self.lam*loss2
        return loss,loss1,loss2

class ObsLossRefactorSigGrad(torch.nn.Module):
    def __init__(self,img_size,patch_size,lossfcn,lam,device):
        super(ObsLossRefactorSigGrad,self).__init__()
        self.device = device
        self.lam = torch.tensor(lam).float().to(device) 
        self.img_size = img_size
        self.patch_size = patch_size
        self.lossfcn = lossfcn
        
    def forward(self,data,net,lossfcn):
        device = self.device

        img_noisy_nosig = data[0]["nosig_real"].to(device)
        img_clean_nosig = data[0]["nosig_image"].to(device)
        img_sigonly = data[1]["sig_window"].to(device)
        img_sigfbp = data[1]["sig_fbp"].to(device)
        img_coordinates_sig = data[1]["sig_coordinates"].to(device)
        img_noisy_sig = img_noisy_nosig+img_sigfbp

        img_recon_sig = net(img_noisy_sig)
        img_recon_nosig = net(img_noisy_nosig)

        mask_sig = torch.zeros_like(img_noisy_sig).bool()
    
        isize = self.img_size
        ix = int(isize[0]/2)
        iy = int(isize[1]/2)

        psize = self.patch_size
        px = int(psize[0]/2)
        py = int(psize[1]/2)
        
        for k in range(len(img_noisy_sig)):
            xx,yy = img_coordinates_sig[k,:]
            mask_sig[k,:,(xx-px):(xx+px),(yy-py):(yy+py)] = True 
 
        img_small_sig = torch.reshape(img_recon_sig[mask_sig],(-1,psize[0],psize[1]))
        img_small_nosig = torch.reshape(img_recon_nosig[mask_sig],(-1,psize[0],psize[1]))
        img_small_sigonly = torch.reshape(img_sigonly[mask_sig],(-1,psize[0],psize[1]))
 
        # Compute loss using passed in lossfcn (with respect to nosig images only)
        loss1 = self.lossfcn(img_recon_nosig,img_clean_nosig)

        # Compute gradients
        img_small_sig_dx = torch.roll(img_small_sig,1,1)-img_small_sig
        img_small_sig_dy = torch.roll(img_small_sig,1,2)-img_small_sig
        img_small_sig_grad = torch.cat((img_small_sig_dx,img_small_sig_dy))

        img_small_nosig_dx = torch.roll(img_small_nosig,1,1)-img_small_nosig
        img_small_nosig_dy = torch.roll(img_small_nosig,1,2)-img_small_nosig
        img_small_nosig_grad = torch.cat((img_small_nosig_dx,img_small_nosig_dy))

        img_small_sigonly_dx = torch.roll(img_small_sigonly,1,1)-img_small_sigonly
        img_small_sigonly_dy = torch.roll(img_small_sigonly,1,2)-img_small_sigonly
        img_small_sigonly_grad = torch.cat((img_small_sigonly_dx,img_small_sigonly_dy))

        # Compute observer Loss
        g1bar = torch.dot(torch.flatten(img_small_sig_grad),torch.flatten(img_small_sigonly_grad))
        g0bar = torch.dot(torch.flatten(img_small_nosig_grad),torch.flatten(img_small_sigonly_grad)) #this doesn't make sense...
        loss2 = g1bar-g0bar

        loss = loss1-self.lam*loss2

        return loss,loss1,loss2
