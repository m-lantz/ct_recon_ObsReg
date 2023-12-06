DATADIR = "/data/realdataFBPsmoothObj_nph4e10/"
bg_image = np.load(PATH+DATADIR+"bg_image_grp1.npy", mmap_mode = "r") #ground truth image, signal absent
nosig_fbp_image  = np.load(PATH+DATADIR+"nosig_128views_fbp_image_grp1.npy", mmap_mode = "r")  #fbp image, signal absent
nosig_fbp_real  = np.load(PATH+DATADIR+"nosig_128views_fbp_real_grp1.npy", mmap_mode = "r")  #noisy fbp image, signal absent
sig_only_fbp_image  = np.load(PATH+DATADIR+"sig_only_fbp_grp1.npy", mmap_mode = "r") #signal only fbp image
sig_coordinates = np.load(PATH+DATADIR+"coords_grp1.npy") #signal location in image

def loadBaseNet():
  '''
  returns the pre-trained Unet with weights for denoiser trained with MSE loss only
  '''
  net = UnetModel(in_chans=1, out_chans=1, num_pool_layers=4, drop_prob=0.0, chans=32)
  net = net.to(device)

  # load pre-trained network weights
  NETNAME = "/models/realdataFBPsmoothObj_nph4e10-pure-denoiser-net.pth"
  net.load_state_dict(torch.load(PATH+NETNAME))
  net.eval();

  return net

def denoise_single_test(net, show_images=True, save_image=False,fname=None):
  '''
  applies the passed in unet model to a single (preset) test image
  displays the results if show_images is True (default)
  saves the reconstructed/output test image if save_image is True
  and fname is specified

  returns the fbp MSE, the denoised MSE, and denoised image
  '''
  DATADIR = "/data/singledata512smoothObj_nph4e10/"

  noisy_image = np.load(PATH+DATADIR+"fbp_128view_noisy.npy")
  truth_image  = np.load(PATH+DATADIR+"fbp_1024view_noiseless.npy")

  t_noisy_image = torch.from_numpy(noisy_image).float().to(device)
  denoised_image = net(t_noisy_image[np.newaxis,np.newaxis,:,:]).detach().cpu().numpy()[0,0,:,:]

  # compute MSE
  f_mse = np.sum(np.square(noisy_image - truth_image))/(512**2)
  print('fbp MSE:      %.8f' % f_mse)

  d_mse = np.sum(np.square(denoised_image - truth_image))/(512**2)
  print('denoised MSE: %.8f' % d_mse)

  test_noisy = noisy_image[240:370,10:200]
  test_recon = denoised_image[240:370,10:200]
  test_clean = truth_image[240:370,10:200]

  if show_images:
    fig, ax = plt.subplots(2, 2, figsize=(15,10))
    ax[0,0].imshow(test_noisy,vmin=0,vmax=0.25, interpolation='nearest')
    ax[0,0].set_title("noisy fbp image")
    ax[0,1].imshow(test_recon,vmin=0,vmax=0.25, interpolation='nearest')
    ax[0,1].set_title("recon")
    ax[1,0].imshow(test_clean,vmin=0,vmax=0.25, interpolation='nearest')
    ax[1,0].set_title("ground truth")
    im = ax[1,1].imshow(np.abs(test_clean-test_recon), interpolation='nearest')
    ax[1,1].set_title("abs(recon-truth)")

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.82, 0.17, 0.01, 0.25])
    fig.colorbar(im, cax=cbar_ax);

  else:
    fig = plt.figure()
    plt.imshow(test_recon,vmin=0,vmax=0.25, interpolation='nearest')

  if save_image:
    fname=f"{PATH}/{fname}.png"
    plt.savefig(fname)
    plt.close()

  return f_mse, d_mse, test_recon

def makeGaussian(size, fwhm = 3, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum (default is 3), which
    can be thought of as an effective radius.

    returns the resulting square Gaussian kernel
    """

    x = np.arange(0, size, 1, float)
    y = x[:,np.newaxis]

    if center is None:
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

def train_epoch(train_images_loader, train_signals_loader, net, lossfcn, obsloss, optimizer, ep, device):
  '''
  takes in dataset of noisy training images and dataset of signals,
  trained denoising network, loss function, obsloss function definition,
  optimizer, and device
  runs a single training epoch for the passed in net
  '''

  epoch=ep
  running_loss = 0.0
  running_loss1 = 0.0
  running_loss2 = 0.0
  for i, data in enumerate(zip(train_images_loader,train_signals_loader)):
      # compute loss + gradients
      optimizer.zero_grad()
      loss,loss1,loss2 = obsloss(data, net, lossfcn)

      loss.backward()
      optimizer.step()

      # print statistics
      running_loss += float(loss)
      running_loss1 += float(loss1)
      running_loss2 += float(loss2)

      if i % 10 == 9:    # print every 10 mini-batches
          print('[%d, %5d] total loss: %.8f, MSE loss: %.8f, Obs loss: -%.8f' %
                  (epoch + 1, i + 1, running_loss / 10, running_loss1 / 10, running_loss2 / 10))
          running_loss = 0.0
          running_loss1 = 0.0
          running_loss2 = 0.0


def train_net(net,sig_num=1000):
  '''
  takes in pre-trained network and the number of signal locations to
  use for training
  sig_num = 1 -> fixed signal location near center
  sig_num = 10 -> 10 random signal locations
  sig_num = 100 -> 100 random signal locations
  sig_num = 1000 -> full set of 1000 signal locations (randomized)
  retrains net for options['epochs'] epochs using observer loss regularization
  '''
  # load training dataset

  nimages = len(bg_image)
  ntrain = nimages
  train_idx = list(range(ntrain))
  sig_alpha = 2*np.ones((nimages,)) #increase signal ampltude by factor of 2 for all images

  train_images_dataset = ImageDataset(bg_image, nosig_fbp_real)
  train_signals_dataset = SignalDataset(sig_only_fbp_image, sig_coordinates, sig_alpha,sig_num=sig_num)


  bs = options['batch_size'] #batchsize (no. of sig absent/sig present)
  train_images_loader = torch.utils.data.DataLoader(train_images_dataset, batch_size=bs, shuffle=True, num_workers=2)
  train_signals_loader = torch.utils.data.DataLoader(train_signals_dataset, batch_size=bs, shuffle=False, num_workers=2) #changed to False

  #Alter residual layer weights prior to retraining. This modifies the network
  #so that it outputs a fraction of the noisy fbp image. This seemed to help
  #push the network away from its initialization and improve in terms of the observer loss.
  alpha = options["alpha"]
  net.res.weight.data[0][0] = alpha+(1.0-alpha)*net.res.weight.data[0][0]
  net.res.weight.data[0][1] = (1.0-alpha)*net.res.weight.data[0][1]
  net.res.requires_grad_(False); #freeze weights on residual layer during training

  img_size = (512,512)
  patch_size = (10,10)
  lossfcn = nn.MSELoss()
  lam = options["lam"]

  # matched filter approach (inner product with signal)
  obsloss = ObsLossRefactorSigInd(img_size,patch_size,lossfcn,lam,device)

  optimizer = torch.optim.Adam(net.parameters(), lr=options['init_lr'])

  for epoch in range(options["epochs"]):
    train_epoch(train_images_loader, train_signals_loader, net, lossfcn, obsloss, optimizer, epoch, device)

