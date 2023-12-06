#Code based on "Validating the use of channels to estimate the ideal linear
#observer" by Gallas and Barret, 2003.

import numpy as np
import scipy
# from scipy import ndimage
# gf = ndimage.gaussian_filter
from math import factorial as fac
from scipy.special import erf
from scipy.special import laguerre

def u(r, a, j):
    return (np.sqrt(2.)/a)*np.exp((-np.pi*r**2.)/a**2.)*laguerre(j)((2.*np.pi*r**2.)/a**2.)

#j = Degree of polynomial, jp = Index

def getchannels(a = 0.5, pc = 1, lgc = 10, sz = 128):
  xar = np.arange(-1. + 1/sz, 1., 2/sz)[:,np.newaxis]*np.ones(sz)
  yar = np.ones(sz)[:,np.newaxis]*np.arange(-1. + 1/sz, 1., 2/sz)
  rar = np.sqrt(xar**2 + yar**2)

  #define channels
  r = rar
  # a = 0.5   #width of the LG channels
  uars = []
  uti = np.zeros([sz,sz])
  #collect the pixel channels first
  for i in range(int(sz/2)-pc,int(sz/2)+pc):
    for j in range(int(sz/2)-pc,int(sz/2)+pc):
        uti.fill(0.)
        uti[i,j] = 1.
        uars.append(uti*1.)
  #collect the Laguerre Gauss channels next
  for d in range(0,lgc):
      b = np.arange(0,d+1)
      uar = u(r, a, d)
      uars.append(1*uar)
  uars = np.array(uars)

  nchan = len(uars)
  # print(nchan)

  # normalize the channels
  normchans = []
  for i in range(nchan):
    newchan = uars[i]*1.
    for nchn in normchans:
        newchan -= ( (newchan*nchn).sum() * nchn)
    newchan /= np.sqrt( (newchan**2).sum() )
    normchans.append(newchan)
  normchans = np.array(normchans)

  uars = normchans*1.
  return uars

def computehcho(uars, sigimages2, bkgimages2, bkgtruth, sigcoords):
  nchan = len(uars)

  sigimages = sigimages2.copy()
  bkgimages = bkgimages2.copy()

  sigimages -= bkgtruth
  bkgimages -= bkgtruth

  nreal = len(sigimages)
  dim = np.shape(sigimages)[1:]

  #center the realizations
  tempsig = np.zeros(2*dim)
  tempbkg = np.zeros(2*dim)
  sigreal = []
  bkgreal = []
  for i in range(nreal):
    tempsig.fill(0.)
    tempbkg.fill(0.)
    x1 = sigcoords[i,0]
    y1 = sigcoords[i,1]
    tempsig[(int(dim[0]/2)-x1):(2*dim[0]-int(dim[0]/2)-x1),(int(dim[1]/2)-y1):(2*dim[1]-int(dim[1]/2)-y1)] = sigimages[i]*1.
    tempbkg[(int(dim[0]/2)-x1):(2*dim[0]-int(dim[0]/2)-x1),(int(dim[1]/2)-y1):(2*dim[1]-int(dim[1]/2)-y1)] = bkgimages[i]*1.
    sigreal.append(1.*tempsig[int(dim[0]/2):(2*dim[0]-int(dim[0]/2)),int(dim[1]/2):(2*dim[1]-int(dim[1]/2))])
    bkgreal.append(1.*tempbkg[int(dim[0]/2):(2*dim[0]-int(dim[0]/2)),int(dim[1]/2):(2*dim[1]-int(dim[1]/2))])
  sigreal=np.array(sigreal)
  bkgreal=np.array(bkgreal)

  trainsig = sigreal[:int(nreal/2)]*1.
  trainbkg = bkgreal[:int(nreal/2)]*1.
  testsig = sigreal[int(nreal/2):]*1.
  testbkg = bkgreal[int(nreal/2):]*1.

  meansig =trainsig.sum(axis=0)/(nreal/2.)
  meanbkg =trainbkg.sum(axis=0)/(nreal/2.)

  meandiff = meansig-meanbkg
  covar = np.zeros([nchan,nchan])
  #use both sig and bkg realizations to estimate covariance
  for i in range(int(nreal/2)):
    ccs = ((uars*(trainsig[i]-meansig)).sum(axis=2)).sum(axis=1)
    covar += np.outer(ccs,ccs)
  for i in range(int(nreal/2)):
    ccs = ((uars*(trainbkg[i]-meanbkg)).sum(axis=2)).sum(axis=1)
    covar += np.outer(ccs,ccs)
  covar/=(nreal-1.)

  cc = ((uars*meandiff).sum(axis=2)).sum(axis=1)
  #compute channelized template
  wcc = np.dot(np.linalg.inv(covar),cc)

  #convert the Hotelling template back to an image
  w = meandiff*0.
  for i in range(nchan):
    w += wcc[i]*uars[i]

  pcscore = 0.
  for i in range(int(nreal/2)):
    sc1 = (testbkg[i]*w).sum()
    for j in range(int(nreal/2)):
        sc2 = (testsig[j]*w).sum()
        if sc2>sc1:
          pcscore += 1.
        if sc2==sc1:
          pcscore += 0.5
  hcho_auc = pcscore/((nreal/2.)**2.)
  # print("trained HO auc: ",auc)

  return w, hcho_auc

def computepixelcho(t_sig_denoised, t_sig_image, t_nosig_denoised, t_nosig_image, t_sig_coordinates, r=1):
  nreal = len(t_sig_denoised)
  sz = np.shape(t_sig_denoised)[1]
  hsz = int(sz/2)
  sigimages = np.zeros([nreal,2*r-1,2*r-1])
  bkgimages = np.zeros([nreal,2*r-1,2*r-1])
  for k in range(nreal):
      [xx,yy] = t_sig_coordinates[k,:].astype('int')
      sigimages[k,:,:] = t_sig_denoised[k,((hsz+xx)-r+1):((hsz+xx)+r),((hsz+yy)-r+1):((hsz+yy)+r)]-t_nosig_image[k,((hsz+xx)-r+1):((hsz+xx)+r),((hsz+yy)-r+1):((hsz+yy)+r)]
      bkgimages[k,:,:] = t_nosig_denoised[k,((hsz+xx)-r+1):((hsz+xx)+r),((hsz+yy)-r+1):((hsz+yy)+r)]-t_nosig_image[k,((hsz+xx)-r+1):((hsz+xx)+r),((hsz+yy)-r+1):((hsz+yy)+r)]

  trainsig = sigimages[:int(nreal/2)]*1.
  trainbkg = bkgimages[:int(nreal/2)]*1.
  testsig = sigimages[int(nreal/2):]*1.
  testbkg = bkgimages[int(nreal/2):]*1.

  meansig =trainsig.sum(axis=0)/(nreal/2.)
  meanbkg =trainbkg.sum(axis=0)/(nreal/2.)

  uars = []
  uti = np.zeros([2*r-1,2*r-1])
  #collect the pixel channels first
  for i in range(2*r-1):
    for j in range(2*r-1):
        uti.fill(0.)
        uti[i,j] = 1.
        uars.append(uti*1.)

  nchan = len(uars)

  meandiff = meansig-meanbkg
  covar = np.zeros([nchan,nchan])
  #use both sig and bkg realizations to estimate covariance
  for i in range(int(nreal/2)):
    ccs = ((uars*(trainsig[i]-meansig)).sum(axis=2)).sum(axis=1)
    covar += np.outer(ccs,ccs)
  for i in range(int(nreal/2)):
    ccs = ((uars*(trainbkg[i]-meanbkg)).sum(axis=2)).sum(axis=1)
    covar += np.outer(ccs,ccs)
  covar/=(nreal-1.)

  cc = ((uars*meandiff).sum(axis=2)).sum(axis=1)
  #compute channelized template
  wcc = np.dot(np.linalg.inv(covar),cc)

  #convert the Hotelling template back to an image
  w = meandiff*0.
  for i in range(nchan):
    w += wcc[i]*uars[i]

  pcscore = 0.
  for i in range(int(nreal/2)):
    sc1 = (testbkg[i]*w).sum()
    for j in range(int(nreal/2)):
        sc2 = (testsig[j]*w).sum()
        if sc2>sc1:
          pcscore += 1.
        if sc2==sc1:
          pcscore += 0.5
  auc = pcscore/((nreal/2.)**2.)
  return w, auc

def computepixelcho_residual(nosig_residual, sig_residual, sig_coordinates, r=1):
  nreal = len(nosig_residual)
  sz = np.shape(nosig_residual)[1]
  hsz = int(sz/2)
  sigimages = np.zeros([nreal,2*r-1,2*r-1])
  bkgimages = np.zeros([nreal,2*r-1,2*r-1])
  for k in range(nreal):
      [xx,yy] = sig_coordinates[k,:].astype('int')
      sigimages[k,:,:] = sig_residual[k,(xx-r+1):(xx+r),(yy-r+1):(yy+r)]
      bkgimages[k,:,:] = nosig_residual[k,(xx-r+1):(xx+r),(yy-r+1):(yy+r)]

  trainsig = sigimages[:int(nreal/2)]*1.
  trainbkg = bkgimages[:int(nreal/2)]*1.
  testsig = sigimages[int(nreal/2):]*1.
  testbkg = bkgimages[int(nreal/2):]*1.

  meansig =trainsig.sum(axis=0)/(nreal/2.)
  meanbkg =trainbkg.sum(axis=0)/(nreal/2.)

  uars = []
  uti = np.zeros([2*r-1,2*r-1])
  #collect the pixel channels first
  for i in range(2*r-1):
    for j in range(2*r-1):
        uti.fill(0.)
        uti[i,j] = 1.
        uars.append(uti*1.)

  nchan = len(uars)

  meandiff = meansig-meanbkg
  covar = np.zeros([nchan,nchan])
  #use both sig and bkg realizations to estimate covariance
  for i in range(int(nreal/2)):
    ccs = ((uars*(trainsig[i]-meansig)).sum(axis=2)).sum(axis=1)
    covar += np.outer(ccs,ccs)
  for i in range(int(nreal/2)):
    ccs = ((uars*(trainbkg[i]-meanbkg)).sum(axis=2)).sum(axis=1)
    covar += np.outer(ccs,ccs)
  covar/=(nreal-1.)

  cc = ((uars*meandiff).sum(axis=2)).sum(axis=1)
  #compute channelized template
  wcc = np.dot(np.linalg.inv(covar),cc)

  #convert the Hotelling template back to an image
  w = meandiff*0.
  for i in range(nchan):
    w += wcc[i]*uars[i]

  pcscore = 0.
  for i in range(int(nreal/2)):
    sc1 = (testbkg[i]*w).sum()
    for j in range(int(nreal/2)):
        sc2 = (testsig[j]*w).sum()
        if sc2>sc1:
          pcscore += 1.
        if sc2==sc1:
          pcscore += 0.5
  auc = pcscore/((nreal/2.)**2.)
  return w, auc

def computehcho_centered(uars, sigimages2, bkgimages2, bkgtruth):
  nchan = len(uars)

  sigimages = sigimages2.copy()
  bkgimages = bkgimages2.copy()

  sigimages -= bkgtruth
  bkgimages -= bkgtruth

  nreal = len(sigimages)
  dim = np.shape(sigimages)[1:]

  #extract 128x128 ROI from 512x512 image
  sigreal = sigimages[:,192:320,192:320]
  bkgreal = bkgimages[:,192:320,192:320]

  trainsig = sigreal[:int(nreal/2)]*1.
  trainbkg = bkgreal[:int(nreal/2)]*1.
  testsig = sigreal[int(nreal/2):]*1.
  testbkg = bkgreal[int(nreal/2):]*1.

  meansig =trainsig.sum(axis=0)/(nreal/2.)
  meanbkg =trainbkg.sum(axis=0)/(nreal/2.)

  meandiff = meansig-meanbkg
  covar = np.zeros([nchan,nchan])
  #use both sig and bkg realizations to estimate covariance
  for i in range(int(nreal/2)):
    ccs = ((uars*(trainsig[i]-meansig)).sum(axis=2)).sum(axis=1)
    covar += np.outer(ccs,ccs)
  for i in range(int(nreal/2)):
    ccs = ((uars*(trainbkg[i]-meanbkg)).sum(axis=2)).sum(axis=1)
    covar += np.outer(ccs,ccs)
  covar/=(nreal-1.)

  cc = ((uars*meandiff).sum(axis=2)).sum(axis=1)
  #compute channelized template
  wcc = np.dot(np.linalg.inv(covar),cc)

  #convert the Hotelling template back to an image
  w = meandiff*0.
  for i in range(nchan):
    w += wcc[i]*uars[i]

  pcscore = 0.
  for i in range(int(nreal/2)):
    sc1 = (testbkg[i]*w).sum()
    for j in range(int(nreal/2)):
        sc2 = (testsig[j]*w).sum()
        if sc2>sc1:
          pcscore += 1.
        if sc2==sc1:
          pcscore += 0.5
  hcho_auc = pcscore/((nreal/2.)**2.)
  # print("trained HO auc: ",auc)

  return w, hcho_auc, wcc

def computehcho_centered_snr(uars, sigimages2, bkgimages2, bkgtruth):
  nchan = len(uars)

  sigimages = sigimages2.copy()
  bkgimages = bkgimages2.copy()

  sigimages -= bkgtruth
  bkgimages -= bkgtruth

  nreal = len(sigimages)
  dim = np.shape(sigimages)[1:]

  #extract 128x128 ROI from 512x512 image
  sigreal = sigimages[:,192:320,192:320]
  bkgreal = bkgimages[:,192:320,192:320]

  trainsig = sigreal[:int(nreal/2)]*1.
  trainbkg = bkgreal[:int(nreal/2)]*1.
  testsig = sigreal[int(nreal/2):]*1.
  testbkg = bkgreal[int(nreal/2):]*1.

  meansig =trainsig.sum(axis=0)/(nreal/2.)
  meanbkg =trainbkg.sum(axis=0)/(nreal/2.)

  meandiff = meansig-meanbkg
  covar = np.zeros([nchan,nchan])
  #use both sig and bkg realizations to estimate covariance
  for i in range(int(nreal/2)):
    ccs = ((uars*(trainsig[i]-meansig)).sum(axis=2)).sum(axis=1)
    covar += np.outer(ccs,ccs)
  for i in range(int(nreal/2)):
    ccs = ((uars*(trainbkg[i]-meanbkg)).sum(axis=2)).sum(axis=1)
    covar += np.outer(ccs,ccs)
  covar/=(nreal-1.)

  cc = ((uars*meandiff).sum(axis=2)).sum(axis=1)
  #compute channelized template
  wcc = np.dot(np.linalg.inv(covar),cc)

  #convert the Hotelling template back to an image
  w = meandiff*0.
  for i in range(nchan):
    w += wcc[i]*uars[i]

  pcscore = 0.
  for i in range(int(nreal/2)):
    sc1 = (testbkg[i]*w).sum()
    for j in range(int(nreal/2)):
        sc2 = (testsig[j]*w).sum()
        if sc2>sc1:
          pcscore += 1.
        if sc2==sc1:
          pcscore += 0.5
  hcho_auc = pcscore/((nreal/2.)**2.)
  # print("trained HO auc: ",auc)

  num = (meandiff*w).sum()**2
  sigma1sq = (1/(int(nreal/2)-1))*np.linalg.norm((testsig-meansig)*w)**2
  sigma2sq = (1/(int(nreal/2)-1))*np.linalg.norm((testbkg-meanbkg)*w)**2
  denom = 0.5*(sigma1sq+sigma2sq)
  snr = np.sqrt(num/denom)

  return w, hcho_auc, wcc, snr

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
