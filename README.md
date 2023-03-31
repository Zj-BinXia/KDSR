# KDSR

This project is the official implementation of 'Knowledge Distillation based Degradation Estimation for Blind Super-Resolution', ICLR2023
> **Knowledge Distillation based Degradation Estimation for Blind Super-Resolution [[Paper](https://arxiv.org/pdf/2211.16928.pdf)] [[Project](https://github.com/Zj-BinXia/KDSR)]**

We provide [Pretrained Models](https://drive.google.com/drive/folders/1_LyZDLu5dNIBaCSu7oB9w9d1SPf1pm8c?usp=sharing) for KDSR-classic (for classic degradation models) and KDSR-GAN (for Real-world SR)

<p align="center">
  <img src="images/method.jpg" width="50%">
</p>

---

##  Dependencies and Installation

- Python >= 3.8 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.10](https://pytorch.org/)

## Dataset Preparation

We use the same training datasets as [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) (DF2K+OST).

### Installation

1. Clone repo

    ```bash
    git clone git@github.com:Zj-BinXia/KDSR.git
    ```

2. If you want to train or test KDSR-GAN (ie, Real-world SR, trained with the same degradation model as Real-ESRGAN)

    ```bash
    cd KDSR-GAN
    ```
    
3. If you want to train or test KDSR-classic (ie, classic degradation models, trained with the isotropic Gaussian Blur or anisotropic Gaussian blur and noises)

    ```bash
    cd KDSR-classic
    ```

**More details please see the README in folder of KDSR-GAN and KDSR-classic** 

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

