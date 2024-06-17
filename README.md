## Introduction

The repository is the code implementation of the paper [A Two-Stage Masked Autoencoder Based Network for Indoor Depth Completion], based on [MAE](https://github.com/facebookresearch/mae) projects.

If you find this project helpful, please give us a star ⭐️, your support is our greatest motivation.

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Model Training](#model-training)
- [Model Testing](#model-testing)
- [Image Prediction](#image-prediction)
- [Acknowledgements](#acknowledgements)
- [Citation](#citation)
- [License](#license)
- [Contact Us](#contact-us)

## Installation

### Dependencies

- Ubuntupyt
- Python 3.7+, recommended 3.7.0
- PyTorch 1.9.0 or higher, recommended 1.9.1+cu111
- CUDA 12.4 or higher, recommended 12.4

### Environment Installation

We recommend using Miniconda for installation. The following command will create a virtual environment named `idc` and install PyTorch.

Note: If you have experience with PyTorch and have already installed it, you can skip to the next section. Otherwise, you can follow these steps to prepare.

<details>

**Step 0**: Install [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/index.html).

**Step 1**: Create a virtual environment named `ttp` and activate it.

```shell
conda create -n ttp python=3.7 -y
conda activate idc
```

**Step 2**: Install [PyTorch2.1.x](https://pytorch.org/get-started/locally/).

Linux:
```shell
pip install torch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 --index-url https://download.pytorch.org/whl/cu111
```

**Step 3**: Install [timm]

```shell
pip install timm=0.4.9
```

**Step 4**: Install other dependencies.

```shell
pip install matplotlib scipy numpy opencv-python pillow typing-extensions=4.2.0
```


</details>

### Install IDC


Download or clone the repository.

```shell
git clone git@github.com:kailaisun/Indoor-Depth-Completion.git
cd Indoor-Depth-Completion
```

## Dataset Preparation

<details>

#### Dataset Download

- Image and label download address: [Matterport3D for Depth Completion](https://github.com/tsunghan-wu/Depth-Completion/blob/master/doc/data.md).
- train : A training dataset of npy files which is concatenated from rgb images and gt depth images for pre-training.
- test : A testing dataset of npy files which is concatenated from rgb images and gt depth images for pre-training.
- train_full : A training dataset of npy files which is concatenated from rgb images, raw depth images and gt depth images for finetuning.
- test_full : A testing dataset of npy files which is concatenated from rgb images, raw depth images and gt depth images for finetuning.

</details>


## Model Training

#### Pretraining

```shell
python tools/train.py configs/TTP/xxx.py  # xxx.py is the configuration file you want to use
```

#### Finetuning

```shell
sh ./tools/dist_train.sh configs/TTP/xxx.py ${GPU_NUM}  # xxx.py is the configuration file you want to use, GPU_NUM is the number of GPUs used
```

## Model Testing

#### Pretraining

```shell
python tools/test.py configs/TTP/xxx.py ${CHECKPOINT_FILE}  # xxx.py is the configuration file you want to use, CHECKPOINT_FILE is the checkpoint file you want to use
```

#### Finetuning

```shell
sh ./tools/dist_test.sh configs/TTP/xxx.py ${CHECKPOINT_FILE} ${GPU_NUM}  # xxx.py is the configuration file you want to use, CHECKPOINT_FILE is the checkpoint file you want to use, GPU_NUM is the number of GPUs used
```

## Image Prediction

#### Single Image Prediction:

```shell
python demo/image_demo_with_cdinferencer.py ${IMAGE_FILE1} ${IMAGE_FILE2} configs/TTP/ttp_sam_large_levircd_infer.py --checkpoint ${CHECKPOINT_FILE} --out-dir ${OUTPUT_DIR}  # IMAGE_FILE is the image file you want to predict, xxx.py is the configuration file, CHECKPOINT_FILE is the checkpoint file you want to use, OUTPUT_DIR is the output path of the prediction result
```

## Acknowledgements

The repository is the code implementation of the paper [Time Travelling Pixels: Bitemporal Features Integration with Foundation Model for Remote Sensing Image Change Detection](https://arxiv.org/abs/2312.16202), based on [MMSegmentation](https://github.com/open-mmlab/mmsegmentation) and [Open-CD](https://github.com/likyoo/open-cd) projects.

## Citation

If you use the code or performance benchmarks of this project in your research, please refer to the following bibtex to cite TTP.

```
@misc{chen2023time,
      title={Time Travelling Pixels: Bitemporal Features Integration with Foundation Model for Remote Sensing Image Change Detection}, 
      author={Keyan Chen and Chengyang Liu and Wenyuan Li and Zili Liu and Hao Chen and Haotian Zhang and Zhengxia Zou and Zhenwei Shi},
      year={2023},
      eprint={2312.16202},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## License

The repository is licensed under the [Apache 2.0 license](LICENSE).

## Contact Us

If you have other questions❓, please contact us in time 👬
