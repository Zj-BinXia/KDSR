# flake8: noqa
import os.path as osp
from basicsr.train import train_pipeline

import kdsrgan.archs
import kdsrgan.data
import kdsrgan.models
import kdsrgan.losses
import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    root_path = osp.abspath(osp.join(__file__, osp.pardir, osp.pardir))
    train_pipeline(root_path)
