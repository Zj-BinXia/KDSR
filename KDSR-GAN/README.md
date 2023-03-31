# KDSR-GAN

This project is the official implementation of 'Knowledge Distillation based Degradation Estimation for Blind Super-Resolution', ICLR2023
> **Knowledge Distillation based Degradation Estimation for Blind Super-Resolution [[Paper](https://arxiv.org/pdf/2211.16928.pdf)] [[Project](https://github.com/Zj-BinXia/KDSR)]**

This is code for KDSR-GAN (for Real-world SR)

<p align="center">
  <img src="images/method.jpg" width="50%">
</p>

---

##  Dependencies and Installation

- Python >= 3.8 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.10](https://pytorch.org/)

### Installation

Install dependent packages


    pip install basicsr 
    pip install -r requirements.txt
    pip install pandas 
    sudo python3 setup.py develop


---

## Training (8 V100 GPUs)

We first train KDSRNet_T (only using L1 loss)

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python3 -m torch.distributed.launch --nproc_per_node=8 --master_port=3309 kdsrgan/train.py -opt options/train_kdsrnet_x4TA.yml --launcher pytorch 
```

Then, we train KDSRNet_S (only using L1 loss and KD loss). **It is notable that modify the ''pretrain_network_TA'' and ''pretrain_network_g'' of options/train_kdsrnet_x4ST.yml to the path of trained KDSRNet_T checkpoint.** Then, we run

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python3  -m torch.distributed.launch --nproc_per_node=8 --master_port=4349 kdsrgan/train.py -opt options/train_kdsrnet_x4ST.yml --launcher pytorch 
```

## :european_castle: Model Zoo

Please download checkpoints from [Google Drive](https://drive.google.com/drive/folders/1QlOz4F9Mtp9DFXoaHYbnMnRSonR9YFJA).
---
## BibTeX

    @InProceedings{xia2022knowledge,
      title={Knowledge Distillation based Degradation Estimation for Blind Super-Resolution},
      author={Xia, Bin and Zhang, Yulun and Wang, Yitong and Tian, Yapeng and Yang, Wenming and Timofte, Radu and Van Gool, Luc},
      journal={ICLR},
      year={2023}
    }

## 📧 Contact

If you have any question, please email `zjbinxia@gmail.com`.



