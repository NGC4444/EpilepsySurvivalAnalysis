# coding=utf-8
# MIT License

# Copyright (c) 2020 Carnegie Mellon University, Auton Lab

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


"""Torch model definitons for the Deep Survival Machines model

This includes definitons for the Torch Deep Survival Machines module.
The main interface is the DeepSurvivalMachines class which inherits
from torch.nn.Module.

Note: NOT DESIGNED TO BE CALLED DIRECTLY!!!

"""
import random
import numpy as np
from PIL import Image
import imp
from matplotlib.style import use
import torch
from torch import nn
import torchvision.transforms as transforms
import os
# from auton_survival.models.dsm import resnet
from auton_survival.models.dsm.load_nii import load_PET
from auton_survival.models.dsm.models.resnet import ResNet, generate_model
# from auton_survival.models.dsm.model import generate_model, parse_opts
# from models import (cnn, C3DNet, resnet, ResNetV2, ResNeXt, ResNeXtV2, WideResNet, PreActResNet,
#         EfficientNet, DenseNet, ShuffleNet, ShuffleNetV2, SqueezeNet, MobileNet, MobileNetV2)
# import resnet
# import torch.utils.model_zoo as mz
__pdoc__ = {}

for clsn in ['DeepSurvivalMachinesTorch',
             'DeepRecurrentSurvivalMachinesTorch',
             'DeepConvolutionalSurvivalMachines']:
  for membr in ['training', 'dump_patches']:

    __pdoc__[clsn+'.'+membr] = False

def create_3DImageRepresentation(tab_indim, layer):
  '''
  layer : number of layer
  '''
  model = generate_model(
            model_depth=10,
            tab_inpdim=tab_indim,
            tab_layer=layer,
            n_classes=128,
            n_input_channels=1,
            shortcut_type='B',
            conv1_t_size=3,
            conv1_t_stride=1,
            no_max_pool=False,
            widen_factor=1.0)
  os.environ["CUDA_VISIBLE_DEVICES"] = "0"
  model = model.cuda(0)
  # torch.distributed.init_process_group('nccl',init_method='file:///home/.../my_file',world_size=1,rank=0)
  # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[0,1,2])
  # model = nn.DataParallel(model, device_ids=[0,1,2])
  return model

def create_representation(inputdim, layers, activation, bias=False):
  r"""Helper function to generate the representation function for DSM.

  Deep Survival Machines learns a representation (\ Phi(X) \) for the input
  data. This representation is parameterized using a Non Linear Multilayer
  Perceptron (`torch.nn.Module`). This is a helper function designed to
  instantiate the representation for Deep Survival Machines.

  .. warning::
    Not designed to be used directly.

  Parameters
  ----------
  inputdim: int
      Dimensionality of the input features.
  layers: list
      A list consisting of the number of neurons in each hidden layer.
  activation: str
      Choice of activation function: One of 'ReLU6', 'ReLU' or 'SeLU'.

  Returns
  ----------
  an MLP with torch.nn.Module with the specfied structure.

  """

  if activation == 'ReLU6':
    act = nn.ReLU6()
  elif activation == 'ReLU':
    act = nn.ReLU()
  elif activation == 'SeLU':
    act = nn.SELU()
  elif activation == 'Tanh':
    act = nn.Tanh()

  modules = []
  prevdim = inputdim

  for hidden in layers:
    modules.append(nn.Linear(prevdim, hidden, bias=bias))
    modules.append(act)
    prevdim = hidden

  return nn.Sequential(*modules)

#高斯噪声
class AddGaussianNoise(object):

    def __init__(self, mean=0.0, variance=1.0, amplitude=1.0,p=1):

        self.mean = mean
        self.variance = variance
        self.amplitude = amplitude
        self.p=p

    def __call__(self, img):
        if random.uniform(0, 1) < self.p:
            img = np.array(img)
            h, w, c = img.shape
            N = self.amplitude * np.random.normal(loc=self.mean, scale=self.variance, size=(h, w, 1))
            N = np.repeat(N, c, axis=2)
            img = N + img
            img[img > 255] = 255                       # 避免有值超过255而反转
            img = Image.fromarray(img.astype('uint8')).convert('RGB')
            return img
        else:
            return img
class DeepSurvivalMachinesTorch(torch.nn.Module):
  """A Torch implementation of Deep Survival Machines model.

  This is an implementation of Deep Survival Machines model in torch.
  It inherits from the torch.nn.Module class and includes references to the
  representation learning MLP, the parameters of the underlying distributions
  and the forward function which is called whenver data is passed to the
  module. Each of the parameters are nn.Parameters and torch automatically
  keeps track and computes gradients for them.

  .. warning::
    Not designed to be used directly.
    Please use the API inferface `dsm.DeepSurvivalMachines` !!!

  Parameters
  ----------
  inputdim: int
      Dimensionality of the input features.
  k: int
      The number of underlying parametric distributions.
  layers: list
      A list of integers consisting of the number of neurons in each
      hidden layer.
  init: tuple
      A tuple for initialization of the parameters for the underlying
      distributions. (shape, scale).
  activation: str
      Choice of activation function for the MLP representation.
      One of 'ReLU6', 'ReLU' or 'SeLU'.
      Default is 'ReLU6'.
  dist: str
      Choice of the underlying survival distributions.
      One of 'Weibull', 'LogNormal'.
      Default is 'Weibull'.
  temp: float
      The logits for the gate are rescaled with this value.
      Default is 1000.
  discount: float
      a float in [0,1] that determines how to discount the tail bias
      from the uncensored instances.
      Default is 1.

  """

  def _init_dsm_layers(self, lastdim):

    if self.dist in ['Weibull']:
      self.act = nn.SELU().cuda(0)
      self.shape = nn.ParameterDict({str(r+1): nn.Parameter(-torch.ones(self.k))
                                     for r in range(self.risks)}).cuda(0)
      self.scale = nn.ParameterDict({str(r+1): nn.Parameter(-torch.ones(self.k))
                                     for r in range(self.risks)}).cuda(0)
    elif self.dist in ['Normal']:
      self.act = nn.Identity().cuda(0)
      self.shape = nn.ParameterDict({str(r+1): nn.Parameter(torch.ones(self.k))
                                     for r in range(self.risks)}).cuda(0)
      self.scale = nn.ParameterDict({str(r+1): nn.Parameter(torch.ones(self.k))
                                     for r in range(self.risks)}).cuda(0)
    elif self.dist in ['LogNormal']:
      self.act = nn.Tanh().cuda(0)
      self.shape = nn.ParameterDict({str(r+1): nn.Parameter(torch.ones(self.k))
                                     for r in range(self.risks)}).cuda(0)
      self.scale = nn.ParameterDict({str(r+1): nn.Parameter(torch.ones(self.k))
                                     for r in range(self.risks)}).cuda(0)
    else:
      raise NotImplementedError('Distribution: '+self.dist+' not implemented'+
                                ' yet.')

    self.gate = nn.ModuleDict({str(r+1): nn.Sequential(
        nn.Linear(lastdim, self.k, bias=False)
        ) for r in range(self.risks)}).cuda(0)

    self.scaleg = nn.ModuleDict({str(r+1): nn.Sequential(
        nn.Linear(lastdim, self.k, bias=True)
        ) for r in range(self.risks)}).cuda(0)

    self.shapeg = nn.ModuleDict({str(r+1): nn.Sequential(
        nn.Linear(lastdim, self.k, bias=True)
        ) for r in range(self.risks)}).cuda(0)

  def __init__(self, inputdim, k, layers=[100], dist='Weibull',
               temp=1000., discount=1.0, optimizer='Adam',
               risks=1):
    super(DeepSurvivalMachinesTorch, self).__init__()

    self.k = k
    self.dist = dist
    self.temp = float(temp)
    self.discount = float(discount)
    self.optimizer = optimizer
    self.risks = risks
    self.Transform =  transforms.Compose([
    # transforms.Resize((224, 224)),
    transforms.RandomVerticalFlip(0.5),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
    AddGaussianNoise(mean=random.uniform(0.5,1.5), variance=0.5, amplitude=random.uniform(0, 45),p = 0.5),
    transforms.RandomVerticalFlip(),
    # transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])])


    if layers is None: layers = []
    self.layers = layers

    if len(layers) == 0: lastdim = inputdim
    else: lastdim = layers[-1]

    self._init_dsm_layers(128)
    # self.embedding = create_representation(inputdim, layers, 'ReLU6')

    #3DResNet
    self.resnet = create_3DImageRepresentation(inputdim, layers)
    print()
    


  def forward(self, x, id,  risk='1', phase = 'train'):
    """The forward function that is called when data is passed through DSM.

    Args:
      x:
        a torch.tensor of the input features.

    """
    # xrep = self.embedding(x.cuda())#模型要放到cuda
    # 加入3DResNet处理Pt
    # used = None
    used, _ = load_PET(id,phase)# nib.Nifti1Image(pet_img[256:,:,:],aff).to_filename('exp1left.nii.gz')#左半边

    used = used.permute(0,3,1,2).unsqueeze(1).cuda(0).double()#1,138,256,256,
    
    xrep, tabpre, imgpre = self.resnet(used, x)
    dim = x.shape[0]
    # return(self.act(self.shapeg[risk](xrep)),
    #        self.act(self.scaleg[risk](xrep)),
    #        self.gate[risk](xrep)/self.temp)
    return(self.act(self.shapeg[risk](xrep))+self.shape[risk].expand(dim, -1),
    self.act(self.scaleg[risk](xrep))+self.scale[risk].expand(dim, -1),
    self.gate[risk](xrep)/self.temp,
    tabpre,
    imgpre)

  def get_shape_scale(self, risk='1'):
    return(self.shape[risk], self.scale[risk])

class DeepRecurrentSurvivalMachinesTorch(DeepSurvivalMachinesTorch):
  """A Torch implementation of Deep Recurrent Survival Machines model.

  This is an implementation of Deep Recurrent Survival Machines model
  in torch. It inherits from `DeepSurvivalMachinesTorch` and replaces the
  input representation learning MLP with an LSTM or RNN, the parameters of the
  underlying distributions and the forward function which is called whenever
  data is passed to the module. Each of the parameters are nn.Parameters and
  torch automatically keeps track and computes gradients for them.

  .. warning::
    Not designed to be used directly.
    Please use the API inferface `dsm.DeepRecurrentSurvivalMachines`!!

  Parameters
  ----------
  inputdim: int
      Dimensionality of the input features.
  k: int
      The number of underlying parametric distributions.
  layers: int
      The number of hidden layers in the LSTM or RNN cell.
  hidden: int
      The number of neurons in each hidden layer.
  init: tuple
      A tuple for initialization of the parameters for the underlying
      distributions. (shape, scale).
  dist: str
      Choice of the underlying survival distributions.
      One of 'Weibull', 'LogNormal'.
      Default is 'Weibull'.
  temp: float
      The logits for the gate are rescaled with this value.
      Default is 1000.
  discount: float
      a float in [0,1] that determines how to discount the tail bias
      from the uncensored instances.
      Default is 1.

  """

  def __init__(self, inputdim, k, typ='LSTM', layers=1,
               hidden=None, dist='Weibull',
               temp=1000., discount=1.0,
               optimizer='Adam', risks=1):
           
    super(DeepSurvivalMachinesTorch, self).__init__()

    self.k = k
    self.dist = dist
    self.temp = float(temp)
    self.discount = float(discount)
    self.optimizer = optimizer
    self.hidden = hidden
    self.layers = layers
    self.typ = typ
    self.risks = risks

    self._init_dsm_layers(hidden)

    if self.typ == 'LSTM':
      self.embedding = nn.LSTM(inputdim, hidden, layers,
                               bias=False, batch_first=True)
    if self.typ == 'RNN':
      self.embedding = nn.RNN(inputdim, hidden, layers,
                              bias=False, batch_first=True,
                              nonlinearity='relu')
    if self.typ == 'GRU':
      self.embedding = nn.GRU(inputdim, hidden, layers,
                              bias=False, batch_first=True)



  def forward(self, x, risk='1'):
    """The forward function that is called when data is passed through DSM.

    Note: As compared to DSM, the input data for DRSM is a tensor. The forward
    function involves unpacking the tensor in-order to directly use the
    DSM loss functions.

    Args:
      x:
        a torch.tensor of the input features.

    """

    x = x.detach().clone()
    inputmask = ~torch.isnan(x[:, :, 0]).reshape(-1)
    x[torch.isnan(x)] = 0

    xrep, _ = self.embedding(x)
    xrep = xrep.contiguous().view(-1, self.hidden)
    xrep = xrep[inputmask]
    xrep = nn.ReLU6()(xrep)

    dim = xrep.shape[0]

    return(self.act(self.shapeg[risk](xrep))+self.shape[risk].expand(dim, -1),
           self.act(self.scaleg[risk](xrep))+self.scale[risk].expand(dim, -1),
           self.gate[risk](xrep)/self.temp)

  def get_shape_scale(self, risk='1'):
    return(self.shape[risk],
           self.scale[risk])

def create_conv_representation(inputdim, hidden,
                               typ='ConvNet', add_linear=True):
  r"""Helper function to generate the representation function for DSM.

  Deep Survival Machines learns a representation (\ Phi(X) \) for the input
  data. This representation is parameterized using a Convolutional Neural
  Network (`torch.nn.Module`). This is a helper function designed to
  instantiate the representation for Deep Survival Machines.

  .. warning::
    Not designed to be used directly.

  Parameters
  ----------
  inputdim: tuple
      Dimensionality of the input image.
  hidden: int
      The number of neurons in each hidden layer.
  typ: str
      Choice of convolutional neural network: One of 'ConvNet'

  Returns
  ----------
  an ConvNet with torch.nn.Module with the specfied structure.

  """

  if typ == 'ConvNet':

    embedding = nn.Sequential(
        nn.Conv2d(1, 6, 3),
        nn.ReLU6(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(6, 16, 3),
        nn.ReLU6(),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.ReLU6(),
    )

  if add_linear:

    dummyx = torch.ones((10, 1) + inputdim)
    dummyout = embedding.forward(dummyx)
    outshape = dummyout.shape

    embedding.add_module('linear', torch.nn.Linear(outshape[-1], hidden))
    embedding.add_module('act', torch.nn.ReLU6())

  return embedding

class DeepConvolutionalSurvivalMachinesTorch(DeepSurvivalMachinesTorch):
  """A Torch implementation of Deep Convolutional Survival Machines model.

  This is an implementation of Deep Convolutional Survival Machines model
  in torch. It inherits from `DeepSurvivalMachinesTorch` and replaces the
  input representation learning MLP with an simple convnet, the parameters of
  the underlying distributions and the forward function which is called whenever
  data is passed to the module. Each of the parameters are nn.Parameters and
  torch automatically keeps track and computes gradients for them.

  .. warning::
    Not designed to be used directly.
    Please use the API inferface
    `dsm.DeepConvolutionalSurvivalMachines`!!

  Parameters
  ----------
  inputdim: tuple
      Dimensionality of the input features. A tuple (height, width).
  k: int
      The number of underlying parametric distributions.
  embedding: torch.nn.Module
      A torch CNN to obtain the representation of the input data.
  hidden: int
      The number of neurons in each hidden layer.
  dist: str
      Choice of the underlying survival distributions.
      One of 'Weibull', 'LogNormal'.
      Default is 'Weibull'.
  temp: float
      The logits for the gate are rescaled with this value.
      Default is 1000.
  discount: float
      a float in [0,1] that determines how to discount the tail bias
      from the uncensored instances.
      Default is 1.

  """

  def __init__(self, inputdim, k,
               embedding=None, hidden=None, dist='Weibull',
               temp=1000., discount=1.0, optimizer='Adam', risks=1):
    super(DeepSurvivalMachinesTorch, self).__init__()

    self.k = k
    self.dist = dist
    self.temp = float(temp)
    self.discount = float(discount)
    self.optimizer = optimizer
    self.hidden = hidden
    self.risks = risks

    self._init_dsm_layers(hidden)

    if embedding is None:
      self.embedding = create_conv_representation(inputdim=inputdim,
                                                  hidden=hidden,
                                                  typ='ConvNet')
    else:
      self.embedding = embedding


  def forward(self, x, risk='1'):
    """The forward function that is called when data is passed through DSM.

    Args:
      x:
        a torch.tensor of the input features.

    """
    xrep = self.embedding(x)

    dim = x.shape[0]
    return(self.act(self.shapeg[risk](xrep))+self.shape[risk].expand(dim, -1),
           self.act(self.scaleg[risk](xrep))+self.scale[risk].expand(dim, -1),
           self.gate[risk](xrep)/self.temp)

  def get_shape_scale(self, risk='1'):
    return(self.shape[risk],
           self.scale[risk])


class DeepCNNRNNSurvivalMachinesTorch(DeepRecurrentSurvivalMachinesTorch):
  """A Torch implementation of Deep CNN Recurrent Survival Machines model.

  This is an implementation of Deep Recurrent Survival Machines model
  in torch. It inherits from `DeepSurvivalMachinesTorch` and replaces the
  input representation learning MLP with an LSTM or RNN, the parameters of the
  underlying distributions and the forward function which is called whenever
  data is passed to the module. Each of the parameters are nn.Parameters and
  torch automatically keeps track and computes gradients for them.

  .. warning::
    Not designed to be used directly.
    Please use the API inferface `dsm.DeepCNNRNNSurvivalMachines`!!

  Parameters
  ----------
  inputdim: tuple
      Dimensionality of the input features. (height, width)
  k: int
      The number of underlying parametric distributions.
  layers: int
      The number of hidden layers in the LSTM or RNN cell.
  hidden: int
      The number of neurons in each hidden layer.
  init: tuple
      A tuple for initialization of the parameters for the underlying
      distributions. (shape, scale).
  dist: str
      Choice of the underlying survival distributions.
      One of 'Weibull', 'LogNormal'.
      Default is 'Weibull'.
  temp: float
      The logits for the gate are rescaled with this value.
      Default is 1000.
  discount: float
      a float in [0,1] that determines how to discount the tail bias
      from the uncensored instances.
      Default is 1.

  """

  def __init__(self, inputdim, k, typ='LSTM', layers=1,
               hidden=None, dist='Weibull',
               temp=1000., discount=1.0,
               optimizer='Adam', risks=1):
    super(DeepSurvivalMachinesTorch, self).__init__()

    self.k = k
    self.dist = dist
    self.temp = float(temp)
    self.discount = float(discount)
    self.optimizer = optimizer
    self.hidden = hidden
    self.layers = layers
    self.typ = typ
    self.risks = risks

    self._init_dsm_layers(hidden)

    self.cnn = create_conv_representation(inputdim, hidden)

    if self.typ == 'LSTM':
      self.rnn = nn.LSTM(hidden, hidden, layers,
                         bias=False, batch_first=True)
    if self.typ == 'RNN':
      self.rnn = nn.RNN(hidden, hidden, layers,
                        bias=False, batch_first=True,
                        nonlinearity='relu')
    if self.typ == 'GRU':
      self.rnn = nn.GRU(hidden, hidden, layers,
                        bias=False, batch_first=True)

  def forward(self, x, risk='1'):
    """The forward function that is called when data is passed through DSM.

    Note: As compared to DSM, the input data for DCRSM is a tensor. The forward
    function involves unpacking the tensor in-order to directly use the
    DSM loss functions.

    Args:
      x:
        a torch.tensor of the input features.

    """

    # Input Mask
    x = x.detach().clone()
    inputmask = ~torch.isnan(x[:, :, 0, 0]).reshape(-1)
    x[torch.isnan(x)] = 0

    # CNN Layer
    xcnn = x.view((-1, 1)+x.shape[2:])
    filteredx = self.cnn(xcnn)

    # RNN Layer
    xrnn = filteredx.view(tuple(x.shape)[:2] + (-1,))
    xrnn, _ = self.rnn(xrnn)
    xrep = xrnn.contiguous().view(-1, self.hidden)

    # Unfolding for DSM
    xrep = xrep[inputmask]
    xrep = nn.ReLU6()(xrep)
    dim = xrep.shape[0]
    return(self.act(self.shapeg[risk](xrep))+self.shape[risk].expand(dim, -1),
           self.act(self.scaleg[risk](xrep))+self.scale[risk].expand(dim, -1),
           self.gate[risk](xrep)/self.temp)

  def get_shape_scale(self, risk='1'):
    return(self.shape[risk],
           self.scale[risk])


