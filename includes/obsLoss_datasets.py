from scipy import ndimage
import torch
import numpy as np

DATADIR = "/data/realdataFBPsmoothObj_nph4e10/"
bg_image = np.load(PATH+DATADIR+"bg_image_grp1.npy", mmap_mode = "r") #ground truth image, signal absent
nosig_fbp_image  = np.load(PATH+DATADIR+"nosig_128views_fbp_image_grp1.npy", mmap_mode = "r")  #fbp image, signal absent
nosig_fbp_real  = np.load(PATH+DATADIR+"nosig_128views_fbp_real_grp1.npy", mmap_mode = "r")  #noisy fbp image, signal absent
sig_only_fbp_image  = np.load(PATH+DATADIR+"sig_only_fbp_grp1.npy", mmap_mode = "r") #signal only fbp image
sig_coordinates = np.load(PATH+DATADIR+"coords_grp1.npy") #signal location in image



class SignalDataset(torch.utils.data.Dataset):
#class for storing signal images
    def __init__(self, sig_only_fbp_image, sig_coordinates, sig_alpha,sig_num='1000'):
        self.sig_only_fbp_image = sig_only_fbp_image
        self.sig_coordinates = sig_coordinates.astype(int)
        self.sig_alpha = sig_alpha.astype('float32') #vector of signal magnification factors
        self.img_size = np.array(nosig_fbp_image.shape[1:])
        self.sig_num = sig_num

    def __len__(self):
        return self.sig_only_fbp_image.shape[0]

    def __getitem__(self,idx):
        '''
        sig_num = the number of signal locations to use during training.
        sig_num = 1 -> use single fixed signal location near center (but not exact center)
        sig_num = 10 -> use 10 random rignal locations
        sig_num = 100 -> use 100 random signal locations
        sig_num = 1000 -> use randomized list of all 1000 signal locations
        '''
        if self.sig_num == 1:
          k = 101 #fixed signal location
        elif self.sig_num == 10:
          k = sig_loc_subset[np.random.randint(0,10)]
        elif self.sig_num ==100:
          k = sig_loc_subset[np.random.randint(0,100)]
        else:
          k = sig_loc_subset[np.random.randint(0,1000)]

        sig_fbp = self.sig_only_fbp_image[k,np.newaxis,:,:].astype('float32')
        coords = self.sig_coordinates[k,:]+(self.img_size/2).astype(int)
        coords_flip = (coords[1],coords[0])

        window = makeGaussian(self.img_size[0], fwhm = 3, center=coords_flip).astype('float32') #gaussian window centered at signal location
        sig_window = sig_fbp*window #sig_fbp image with streaking artifacts suppressed

        data = {}
        data["sig_template"] = self.sig_alpha[idx]*sig_window #template "shat" used in observer regularizer
        data["sig_fbp"] = self.sig_alpha[idx]*sig_fbp #sig fbp image "s" to be added to noisy fbp image in observer regularizer
        data["sig_coordinates"] = coords
        data["window"] = window
        data["sig_window"]=sig_window

        return data

class SignalDatasetL(torch.utils.data.Dataset):
#class for storing signal images
    def __init__(self, sig_only_fbp_image, sig_coordinates, sig_alpha,sig_num='1000'):
        self.sig_only_fbp_image = sig_only_fbp_image
        self.sig_coordinates = sig_coordinates.astype(int)
        self.sig_alpha = sig_alpha.astype('float32') #vector of signal magnification factors
        self.img_size = np.array(nosig_fbp_image.shape[1:])
        self.sig_num = sig_num

    def __len__(self):
        return self.sig_only_fbp_image.shape[0]

    def __getitem__(self,idx):
        '''
        sig_num = the number of signal locations to use during training.
        sig_num = 1 -> use single fixed signal location near center (but not exact center)
        sig_num = 10 -> use 10 random rignal locations
        sig_num = 100 -> use 100 random signal locations
        sig_num = 1000 -> use randomized list of all 1000 signal locations
        '''
        if self.sig_num == 1:
          k = 101 #fixed signal location
        elif self.sig_num == 10:
          k = sig_loc_subset[np.random.randint(0,10)]
        elif self.sig_num ==100:
          k = sig_loc_subset[np.random.randint(0,100)]
        else:
          k = sig_loc_subset[np.random.randint(0,1000)]

        sig_fbp = self.sig_only_fbp_image[k,np.newaxis,:,:].astype('float32')
        coords = self.sig_coordinates[k,:]+(self.img_size/2).astype(int)
        coords_flip = (coords[1],coords[0])

        window = makeGaussian(self.img_size[0], fwhm = 3, center=coords_flip).astype('float32') #gaussian window centered at signal location
        sig_window = sig_fbp*window #sig_fbp image with streaking artifacts suppressed
        sig_window = -ndimage.laplace(sig_window) #lapacian filer of sig_window

        data = {}
        data["sig_template"] = self.sig_alpha[idx]*sig_window #template "shat" used in observer regularizer
        data["sig_fbp"] = self.sig_alpha[idx]*sig_fbp #sig fbp image "s" to be added to noisy fbp image in observer regularizer
        data["sig_coordinates"] = coords
        data["window"] = window
        data["sig_window"]=sig_window

        return data

class ImageDataset(torch.utils.data.Dataset):
#class for storing noisy/clean training pairs
    def __init__(self, bg_image, nosig_fbp_real):
        self.bg_image = bg_image
        self.nosig_fbp_real = nosig_fbp_real
        self.img_size = np.array(bg_image.shape[1:])

    def __len__(self):
        return self.bg_image.shape[0]

    def __getitem__(self,idx):
        bg_image = self.bg_image[idx,np.newaxis,:,:].astype('float32')
        nosig_fbp_real = self.nosig_fbp_real[idx,np.newaxis,:,:].astype('float32')

        data = {}
        data["nosig_image"] = bg_image #ground truth
        data["nosig_real"] = nosig_fbp_real #noisy fbp image (no signal)
        return data

