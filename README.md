# KDSR

This project is the official implementation of 'Knowledge Distillation based Degradation Estimation for Blind Super-Resolution', ICLR2023
> **Knowledge Distillation based Degradation Estimation for Blind Super-Resolution [[Paper](https://arxiv.org/pdf/2211.16928.pdf)] [[Project](https://github.com/Zj-BinXia/KDSR)]**

We provide [Pretrained Models](https://drive.google.com/drive/folders/1_LyZDLu5dNIBaCSu7oB9w9d1SPf1pm8c?usp=sharing) for KDSR-classic (for classic degradation models) and KDSR-GAN (for Real-world SR)

<p align="center">
  <img src="images/method.jpg" width="50%">
</p>

- **August 28, 2023:** For real-world SR tasks, we released a [KDSR-GANV2] that is more focused on perception rather than the distortion， which can be used to super-resolve AIGC generated images.
| \multirow{2}[2]{*}{Methods} | \multirow{2}[2]{*}{Parms (M)} | \multirow{2}[2]{*}{FLOPs(G)} | \multirow{2}[2]{*}{Runtime (ms)} | \multicolumn{3}{c|}{AIM2019} | \multicolumn{3}{c}{NTIRE2020} |
|-----------------------------|-------------------------------|------------------------------|----------------------------------|------------------------------|-------------------------------|
|                             |                               |                              |                                  | LPIPS$\downarrow$            | PSNR$\uparrow$                | SSIM$\uparrow$  | LPIPS$\downarrow$ | PSNR$\uparrow$ | SSIM$\uparrow$  |
| ESRGAN                      | 16.69                         | 871.25                       | 236.04                           | 0.5558                       | 23.17                         | 0.6192          | 0.5938            | 21.14          | 0.3119          |
| % DnCNN+DCLS                | 19.70                         | -                            | 192.83                           | 0.5362                       | 24.20                         | 0.6810          | 0.4279            | 28.63          | 0.7851          |
| BSRGAN                      | 16.69                         | 871.25                       | 236.04                           | 0.4048                       | 24.20                         | 0.6904          | 0.3691            | 26.75          | 0.7386          |
| Real-ESRGAN                 | 16.69                         | 871.25                       | 236.04                           | 0.3956                       | 23.89                         | 0.6892          | 0.3471            | 26.40          | 0.7431          |
| MM-RealSR                   | 26.13                         | 930.54                       | 290.64                           | 0.3948                       | 23.45                         | 0.6889          | 0.3446            | 25.19          | 0.7404          |
| KDSR$_{s}$-GAN (Ours)       | 18.85                         | 640.84                       | 154.62                           | \textbf{0.3758}              | \textbf{24.22}                | \textbf{0.7038} | \textbf{0.3198}   | \textbf{27.12} | \textbf{0.7614} |
- **Jan 28, 2023:** Training&Testing codes and pre-trained models are released!

---

##  Dependencies and Installation

- Python >= 3.8 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux) or [Miniconda](https://docs.conda.io/en/latest/miniconda.html))
- [PyTorch >= 1.10](https://pytorch.org/)


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

