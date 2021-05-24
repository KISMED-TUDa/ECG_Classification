from typing import Tuple, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
import numpy as np
from torchaudio.transforms import Spectrogram
import os
import gzip
import pickle

class ChallengeDataset(Dataset):
    """
    This class implements the dataset from "Wettbewerb Künstliche Intelligenz in der Medizin" at TU Darmstadt"
    """