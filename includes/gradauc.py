#compute AUC and SNR using gradient of signal as a single channel

import numpy as np
import scipy
# from scipy import ndimage
# gf = ndimage.gaussian_filter
from math import factorial as fac
from scipy.special import erf
from scipy.special import laguerre

def gradim(im):
   temp = im
   xgrad = im*0.
   xgrad[:-1] = temp[1:] - temp[:-1]
   xgrad[-1] =  -1.0* temp[-1]
   ygrad = im*0.
   ygrad[:,:-1] = temp[:,1:] - temp[:,:-1]
   ygrad[:,-1] =  -1.0* temp[:,-1]
   return np.concatenate((xgrad,ygrad))

# def mdiv(xgrad,ygrad):
#    divim = xgrad*1.
#    shp = [xgrad.shape[0] + 2,xgrad.shape[1]]
#    xgradp=zeros(shp,float64)
#    xgradp[1:-1] = xgrad*1.
#    shp = [ygrad.shape[0],ygrad.shape[1]+2]
#    ygradp=zeros(shp,float64)
#    ygradp[:,1:-1] = ygrad*1.
#    divim.fill(0.)
#    divim = xgradp[:-2] - xgradp[1:-1] + ygradp[:,:-2] - ygradp[:,1:-1]
#    return divim


# disk = xar*0.
# x0 = 0.0
# y0 = 0.0
# rad = 0.002
# disk[(xar-x0)**2. + (yar-y0)**2.<= rad] = 1.0

# nstd = 2.0
# noisemean =  zeros([nx,ny])
# noisevar = nstd**2 * ones([nx,ny])
# whotel = disk/noisevar

# snr1 = sqrt( (whotel*disk).sum() )
# print "snr1: ", snr1


# def gradchan(w):
#    xgch,ygch=gradim(disk)
#    xgf,ygf=gradim(gfilt(disk,w))
#    xgi,ygi = gradim(gfilt(noisevar*gfilt(mdiv(xgch,ygch),w),w))
#    chvar = (xgch*xgi + ygch*ygi).sum()
#    chsig = (xgch*xgf + ygch*ygf).sum()
#    whotel = chsig/chvar
#    snr = sqrt( (whotel*chsig).sum() )
# #   print "sig part: %f and noise part: %f"%(chsig**2.,chvar)
#    print "gradch snr w=%f: %f and auc: %f"%(w,snr,0.5+0.5*erf(snr*snr/2))

# def sigchan(w):
#    ch=1.*disk
#    gf= gfilt(disk,w)
#    chvar = (ch*gfilt(noisevar*gfilt(ch,w),w)).sum()
#    chsig = (ch*gf).sum()
#    whotel = chsig/chvar
#    snr = sqrt( (whotel*chsig).sum() )
# #   print "sig part: %f and noise part: %f"%(chsig**2.,chvar)
#    print "sigch snr w=%f: %f and auc: %f"%(w,snr,0.5+0.5*erf(snr*snr/2))

#compute auc using gradient of signal as template
def gauc(sigimages2, bkgimages2, bkgtruth, sigonly2, roi):
  sigimages = sigimages2.copy()
  bkgimages = bkgimages2.copy()

  sigimages -= bkgtruth
  bkgimages -= bkgtruth

  nreal = len(sigimages)
  dim = np.shape(sigimages)[1:]

  #extract ROI from 512x512 image
  i1 = 256-int(roi[0]/2)
  i2 = 256+int(roi[1]/2)+1
  testsig = sigimages[:,i1:i2,i1:i2]
  testbkg = bkgimages[:,i1:i2,i1:i2]
  sigonly = sigonly2[i1:i2,i1:i2] 
  w = gradim(sigonly)

  # trainsig = sigreal[:int(nreal/2)]*1.
  # trainbkg = bkgreal[:int(nreal/2)]*1.
  # testsig = sigreal[int(nreal/2):]*1.
  # testbkg = bkgreal[int(nreal/2):]*1.

  # meansig =trainsig.sum(axis=0)/(nreal/2.)
  # meanbkg =trainbkg.sum(axis=0)/(nreal/2.)

  # meandiff = meansig-meanbkg
  # covar = np.zeros([nchan,nchan])
  # #use both sig and bkg realizations to estimate covariance
  # for i in range(int(nreal/2)):
  #   ccs = ((uars*(trainsig[i]-meansig)).sum(axis=2)).sum(axis=1)
  #   covar += np.outer(ccs,ccs)
  # for i in range(int(nreal/2)):
  #   ccs = ((uars*(trainbkg[i]-meanbkg)).sum(axis=2)).sum(axis=1)
  #   covar += np.outer(ccs,ccs)
  # covar/=(nreal-1.)

  # cc = ((uars*meandiff).sum(axis=2)).sum(axis=1)
  # #compute channelized template
  # wcc = np.dot(np.linalg.inv(covar),cc)

  # #convert the Hotelling template back to an image
  # w = meandiff*0.
  # for i in range(nchan):
  #   w += wcc[i]*uars[i]

  pcscore = 0.
  for i in range(int(nreal/2)):
    sc1 = (gradim(testbkg[i])*w).sum()
    for j in range(int(nreal/2)):
        sc2 = (gradim(testsig[j])*w).sum()
        if sc2>sc1:
          pcscore += 1.
        if sc2==sc1:
          pcscore += 0.5
  grad_auc = pcscore/((nreal/2.)**2.)
  # print("trained HO auc: ",auc)

  #compute test score stats
  teststat_bkg = np.zeros(int(nreal/2))
  teststat_sig = np.zeros(int(nreal/2))
  for i in range(int(nreal/2)):
    teststat_bkg[i] = (gradim(testbkg[i])*w).sum()
  for i in range(int(nreal/2)):
    teststat_sig[i] = (gradim(testsig[i])*w).sum()

  return grad_auc, teststat_bkg, teststat_sig

#compute auc using signal as template
def sigauc(sigimages2, bkgimages2, bkgtruth, sigonly2, roi):
  sigimages = sigimages2.copy()
  bkgimages = bkgimages2.copy()

  sigimages -= bkgtruth
  bkgimages -= bkgtruth

  nreal = len(sigimages)
  dim = np.shape(sigimages)[1:]

  #extract 10x10 ROI from 512x512 image
  i1 = 256-int(roi[0]/2)
  i2 = 256+int(roi[1]/2)+1
  testsig = sigimages[:,i1:i2,i1:i2]
  testbkg = bkgimages[:,i1:i2,i1:i2]
  w = sigonly2[i1:i2,i1:i2] 

  pcscore = 0.
  for i in range(int(nreal/2)):
    sc1 = (testbkg[i]*w).sum()
    for j in range(int(nreal/2)):
        sc2 = (testsig[j]*w).sum()
        if sc2>sc1:
          pcscore += 1.
        if sc2==sc1:
          pcscore += 0.5
  sig_auc = pcscore/((nreal/2.)**2.)
  # print("trained HO auc: ",auc)

  #compute test score stats
  teststat_bkg = np.zeros(int(nreal/2))
  teststat_sig = np.zeros(int(nreal/2))
  for i in range(int(nreal/2)):
    teststat_bkg[i] = (testbkg[i]*w).sum()
  for i in range(int(nreal/2)):
    teststat_sig[i] = (testsig[i]*w).sum()

  return sig_auc, teststat_bkg, teststat_sig

#compute SNR using signal as template
def sigsnr(sigimages2, bkgimages2, bkgtruth, sigonly2, roi):
  sigimages = sigimages2.copy()
  bkgimages = bkgimages2.copy()

  sigimages -= bkgtruth
  bkgimages -= bkgtruth

  nreal = len(sigimages)
  dim = np.shape(sigimages)[1:]

  #extract center ROI from 512x512 image
  i1 = 256-int(roi[0]/2)
  i2 = 256+int(roi[1]/2)+1
  testsig = sigimages[:,i1:i2,i1:i2]
  testbkg = bkgimages[:,i1:i2,i1:i2]
  w = sigonly2[i1:i2,i1:i2] 

  meansig = testsig.sum(axis=0)/(nreal)
  meanbkg = testbkg.sum(axis=0)/(nreal)
  meandiff = meansig-meanbkg

  num = (meandiff*w).sum()**2
  sigma1sq = (1/(nreal-1))*np.linalg.norm((testsig-meansig)*w)**2
  sigma2sq = (1/(nreal-1))*np.linalg.norm((testbkg-meanbkg)*w)**2
  denom = 0.5*(sigma1sq+sigma2sq)

  return np.sqrt(num/denom)

#compute SNR using signal gradient as template
def gsnr(sigimages2, bkgimages2, bkgtruth, sigonly2, roi):
  sigimages = sigimages2.copy()
  bkgimages = bkgimages2.copy()

  sigimages -= bkgtruth
  bkgimages -= bkgtruth

  nreal = len(sigimages)
  dim = np.shape(sigimages)[1:]

  #extract ROI from 512x512 image
  i1 = 256-int(roi[0]/2)
  i2 = 256+int(roi[1]/2)+1
  testsig = sigimages[:,i1:i2,i1:i2]
  testbkg = bkgimages[:,i1:i2,i1:i2]
  sigonly = sigonly2[i1:i2,i1:i2] 
  w = gradim(sigonly)

  meansig = testsig.sum(axis=0)/nreal
  meanbkg = testbkg.sum(axis=0)/nreal
  meandiff = meansig-meanbkg

  gsig = np.zeros(np.append(nreal,w.shape))
  gbkg = np.zeros(np.append(nreal,w.shape))
  for i in range(nreal):
    gsig[i,:] = gradim(testsig[i,:]-meansig)
    gbkg[i,:] = gradim(testbkg[i,:]-meansig)

  num = (gradim(meandiff)*w).sum()**2
  sigma1sq = (1.0/(nreal-1))*(np.linalg.norm(gsig*w))**2
  sigma2sq = (1.0/(nreal-1))*(np.linalg.norm(gbkg*w))**2
  denom = 0.5*(sigma1sq+sigma2sq)

  return np.sqrt(num/denom)

  