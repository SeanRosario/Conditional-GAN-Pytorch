from __future__ import print_function
import os
import time
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
from torchvision.transforms import ToPILImage
from IPython.display import Image
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline

### load project files
import models_cgan as models
from models_cgan import weights_init


# In[7]:

class args:
	dataRoot='./posters_resized'
	workers=12
	batchSize=128
	imageSize=64
	nz=100
	ngf=64
	ndf=64
	niter=80
	lr=0.0002
	beta1=0.5
	cuda=True
	ngpu=1
	netG= ''
	netD= ''
	outDir='./results'
	model=1
	d_labelSmooth=0.1      # 0.25 from imporved-GANpaper
	n_extra_layers_d=0
	n_extra_layers_g=1     #in the sense that generator should be more powerful
	binary = False


# In[8]:

try:
    os.makedirs(args.outDir)
except OSError:
    pass


# In[9]:

#args = parser.parse_args()
#args = parser.parse_args(arg_list)
print(args)

try:
    os.makedirs(args.outDir)
except OSError:
    pass

args.manualSeed = random.randint(1,10000) # fix seed, a scalar
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not args.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    
nc = 8
ngpu = args.ngpu
nz = args.nz
ngf = args.ngf
ndf = args.ndf
n_extra_d = args.n_extra_layers_d
n_extra_g = args.n_extra_layers_g

dataset = dset.ImageFolder(
    root=args.dataRoot,
    transform=transforms.Compose([
            transforms.Scale(args.imageSize),
            # transforms.CenterCrop(args.imageSize),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)), # bring images to (-1,1)
        ])
)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batchSize,
                                         shuffle=True, num_workers=args.workers)

# load models
if args.model == 0:
    netG = models._netG_0(ngpu, nz, 3, ngf)
    netD = models._netD_0(ngpu, nz, nc, ndf)
elif args.model == 1:
    netG = models._netG_1(ngpu, 105, 3, ngf, n_extra_g)
    netD = models._netD_1(ngpu, 105, nc, ndf, n_extra_d)
elif args.model == 2:
    netG = models._netG_2(ngpu, nz, 3, ngf)
    netD = models._netD_2(ngpu, nz, nc, ndf)

netG.apply(weights_init)
if args.netG != '':
    netG.load_state_dict(torch.load(args.netG))
print(netG)

netD.apply(weights_init)
if args.netD != '':
    netD.load_state_dict(torch.load(args.netD))
print(netD)


# In[219]:

criterion = nn.BCELoss()
criterion_MSE = nn.MSELoss()

input = torch.FloatTensor(args.batchSize, nc, args.imageSize, args.imageSize)
noise = torch.FloatTensor(args.batchSize, nz, 1, 1)
if args.binary:
    bernoulli_prob = torch.FloatTensor(args.batchSize, nz, 1, 1).fill_(0.5)
    fixed_noise = torch.bernoulli(bernoulli_prob)
else:
    fixed_noise = torch.FloatTensor(args.batchSize, nz, 1, 1).normal_(0, 1)
label = torch.FloatTensor(args.batchSize)
real_label = 1
fake_label = 0


input = Variable(input)
label = Variable(label)
noise = Variable(noise)
fixed_noise = Variable(fixed_noise)


# setup argsimizer
optimizerD = optim.Adam(netD.parameters(), lr=args.lr, betas=(args.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=args.lr, betas=(args.beta1, 0.999))




# In[155]:

def concat_channel(images,labels):
    new_images = []
    for image,label in zip(images,labels):
        a = np.zeros([5,64,64])
        a[label-1] += 1
        new_image = np.concatenate([image.numpy(),a])
        
        new_images.append(new_image)
        
    new_images = np.stack(new_images)
    return torch.from_numpy(new_images).float()


# In[179]:

def newconcat(images, labels):
    y_onehot = torch.FloatTensor(args.batchSize, 5)
    #print (label.unsqueeze(1).size())
    try:
        y_onehot.zero_().scatter_(1, labels.unsqueeze(1), 1)
    except:
        y_onehot.zero_().scatter_(1, labels, 1)
    y_onehot = torch.unsqueeze(torch.unsqueeze(y_onehot, 2),3).expand(args.batchSize, 5, args.imageSize,
                                                                      args.imageSize)

    return torch.cat((images, y_onehot),1)


# In[192]:

def newconcat_noise(images, labels):
    y_onehot = torch.FloatTensor(args.batchSize, 5)
    #print (label.unsqueeze(1).size())
    try:
        y_onehot.zero_().scatter_(1, labels.unsqueeze(1), 1)
    except:
        y_onehot.zero_().scatter_(1, labels, 1)
    #y_onehot = torch.unsqueeze(torch.unsqueeze(y_onehot, 2),3).expand(args.batchSize, 5, args.imageSize,
    #args.imageSize)

    return torch.cat((images, y_onehot),1)





# In[220]:

for epoch in range(args.niter):
    for i, data in enumerate(dataloader, 0):
        print("Epoch start")
        start_iter = time.time()
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real

        netD.zero_grad()
        images,labels= data[0],data[1]
        images = newconcat(images,labels)
        data = images,labels
        

        
        real_cpu = images
        if real_cpu.size(0)!=args.batchSize:
            print ("2")
            continue
        
        batch_size = args.batchSize

        input.data.resize_(real_cpu.size()).copy_(real_cpu)
        label.data.resize_(batch_size).fill_(real_label - args.d_labelSmooth) # use smooth label for discriminator
        output = netD(input)

        
        errD_real = criterion(output, label)
        errD_real.backward()
        D_x = output.data.mean()
        
        # train with fake
        noise.data.resize_(batch_size, nz, 1, 1)
        if args.binary:
            bernoulli_prob.resize_(noise.data.size())
            noise.data.copy_(2*(torch.bernoulli(bernoulli_prob)-0.5))
        else:
            noise.data.normal_(0, 1)

        try:
            noise = Variable(newconcat_noise(noise.data,labels))

            fake = netG(noise)
            fake = Variable(newconcat(fake.data,labels))
        except:
            print ("3")
            continue
        
        label.data.fill_(fake_label)
        output = netD(fake.detach()) # add ".detach()" to avoid backprop through G

        errD_fake = criterion(output, label)
        errD_fake.backward() # gradients for fake/real will be accumulated
        D_G_z1 = output.data.mean()
        errD = errD_real + errD_fake
        optimizerD.step() # .step() can be called once the gradients are computed

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.data.fill_(real_label) # fake labels are real for generator cost
        output = netD(fake)
        errG = criterion(output, label)
        errG.backward(retain_variables=True) # True if backward through the graph for the second time
        if args.model == 2: # with z predictor
            errG_z = criterion_MSE(z_prediction, noise)
            errG_z.backward()
        D_G_z2 = output.data.mean()
        optimizerG.step()
        
        end_iter = time.time()
        print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f Elapsed %.2f s'
              % (epoch, args.niter, i, len(dataloader),
                 errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2, end_iter-start_iter))
        if i % 100 == 0:
            # the first 64 samples from the mini-batch are saved.
            real_cpu
            vutils.save_image(real_cpu[0:64,:3,:,:],
                    '%s/real_samples%03d.png' % (args.outDir, epoch), nrow=8)
            try:
                fake = netG(Variable(newconcat_noise(fixed_noise.data,labels)))
            except:
                continue
            vutils.save_image(fake.data[0:64,:3,:,:],
                    '%s/fake_samples_epoch_%03d.png' % (args.outDir, epoch), nrow=8)
    if epoch % 1 == 0:
        # do checkpointing
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (args.outDir, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (args.outDir, epoch))



# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



