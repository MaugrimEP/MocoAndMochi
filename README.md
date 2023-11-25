## MoCo: Momentum Contrast for Unsupervised Visual Representation Learning
## Mochi: Hard Negative Mixing for Contrastive Learning

This repository is based on facebookresearch/moco, with modifications.


```
pip install numpy torch blobfile tqdm pyYaml pillow jaxtyping beartype pytorch-lightning omegaconf hydra-core wandb
```

<p align="center">
  <img src="https://user-images.githubusercontent.com/11435359/71603927-0ca98d00-2b14-11ea-9fd8-10d984a2de45.png" width="300">
</p>
<p align="center">
  <img src="https://europe.naverlabs.com/wp-content/uploads/2020/10/hard-negative-mixing.png" width="300">
</p>

This is a PyTorch implementation of the [MoCo paper](https://arxiv.org/abs/1911.05722):
```
@Article{he2019moco,
  author  = {Kaiming He and Haoqi Fan and Yuxin Wu and Saining Xie and Ross Girshick},
  title   = {Momentum Contrast for Unsupervised Visual Representation Learning},
  journal = {arXiv preprint arXiv:1911.05722},
  year    = {2019},
}
```
It also includes the implementation of the [MoCo v2 paper](https://arxiv.org/abs/2003.04297):
```
@Article{chen2020mocov2,
  author  = {Xinlei Chen and Haoqi Fan and Ross Girshick and Kaiming He},
  title   = {Improved Baselines with Momentum Contrastive Learning},
  journal = {arXiv preprint arXiv:2003.04297},
  year    = {2020},
}
```
And an implementation of the [Mochi paper](https://europe.naverlabs.com/research/computer-vision/mochi/):
```
@InProceedings{kalantidis2020hard,
  author = {Kalantidis, Yannis and Sariyildiz, Mert Bulent and Pion, Noe and Weinzaepfel, Philippe and Larlus, Diane},
  title = {Hard Negative Mixing for Contrastive Learning},
  booktitle = {Neural Information Processing Systems (NeurIPS)},
  year = {2020}
}
```
For more details on data, training, loading and results, please see the [original repo](https://github.com/facebookresearch/moco).

I use Hydra configuration and not the original argparse.

New main file for pretraining is `lightning_main_pretraining.py` and new file for fine-tuning is `lightning_main_finetuning.py`.

This implementation with Pytorch Lightning should support ddp as well as single GPU implementation.

### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.

### See Also
* [moco.tensorflow](https://github.com/ppwwyyxx/moco.tensorflow): A TensorFlow re-implementation.
* [Colab notebook](https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb): CIFAR demo on Colab GPU.
