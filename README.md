# Knowledge Distillation based Degradation Estimation for Blind Super-Resolution （ICLR2023）


[Paper](https://arxiv.org/pdf/2211.16928.pdf) | [Project Page](https://github.com/Zj-BinXia/KDSR) | [pretrained models](https://drive.google.com/drive/folders/1_LyZDLu5dNIBaCSu7oB9w9d1SPf1pm8c?usp=sharing)

#### News

- **August 28, 2023:** For real-world SR tasks, we released a pretrained model [KDSR-GANV2](https://drive.google.com/file/d/1plvMt7VrOY9YLbWrpchOzi6t1wcqkzBl/view?usp=sharing) and [training files](KDSR-GAN/options/train_kdsrgan_x4STV2.yml) that is more focused on perception rather than distortion.
  


- **Jan 28, 2023:** Training&Testing codes and pre-trained models are released!

<hr />


> **Abstract:** *Blind image super-resolution (Blind-SR) aims to recover a high-resolution (HR) image from its corresponding low-resolution (LR) input image with unknown degradations. Most of the existing works design an explicit degradation estimator for each degradation to guide SR. However, it is infeasible to provide concrete labels of multiple degradation combinations (\eg, blur, noise, jpeg compression) to supervise the degradation estimator training. In addition, these special designs for certain degradation, such as blur, impedes the models from being generalized to handle different degradations. To this end, it is necessary to design an implicit degradation estimator that can extract discriminative degradation representation for all degradations without relying on the supervision of degradation ground-truth. In this paper, we propose a Knowledge Distillation based Blind-SR network (KDSR). It consists of a knowledge distillation based implicit degradation estimator network (KD-IDE) and an efficient SR network. To learn the KDSR model, we first train a teacher network: KD-IDE$_{T}$. It takes paired HR and LR patches as inputs and is optimized with the SR network jointly. Then, we further train a student network KD-IDE$_{S}$, which only takes LR images as input and learns to extract the same implicit degradation representation (IDR) as KD-IDE$_{T}$. In addition, to fully use extracted IDR, we design a simple, strong, and efficient IDR based dynamic convolution residual block (IDR-DCRB) to build an SR network. We conduct extensive experiments under classic and real-world degradation settings. The results show that KDSR achieves SOTA performance and can generalize to various degradation processes.* 


<p align="center">
  <img src="images/method.jpg" width="80%">
</p>



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

