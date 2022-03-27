import torch 
import torch.backends.cudnn
import torch.nn as nn
import torch.nn.functional as F 
from torch.nn.parameter import Parameter
import torch.optim as optim 
import numpy as np 
import scipy 
import scipy.io as sio 
from torch import Tensor 
from numpy import ndarray
import random
import logging  
from typing import Optional, Union, List, Dict, Tuple 
import itertools
import math 
from datetime import datetime, date, timedelta
from tqdm import tqdm 
from pprint import pprint 

import dgl
import dgl.function as dglfn 
import dgl.nn as dglnn
import dgl.nn.functional as dglF 

DEVICE = 'cuda:2'

IntTensor = FloatTensor = Tensor 
IntArray = FloatArray = ndarray 

logging.basicConfig(format='%(asctime)s [%(levelname)s] %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[
                        logging.FileHandler("./log.log", 'w', encoding='utf-8'),
                        logging.StreamHandler()
                    ],
                    level=logging.INFO)


def seed_all(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def to_device(x):
    return x.to(device=torch.device(DEVICE))
