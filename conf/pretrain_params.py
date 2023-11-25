from dataclasses import dataclass

import torchvision.models as models
from omegaconf import SI

model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)


@dataclass
class PretrainParams:
    data: str = r"/home/2022022/PARTAGE/Imagenet-100"
    r"""
    C:\Users\tmayet\MobaXterm\home\Imagenet-100
    /home/2022022/PARTAGE/Imagenet-100
    """
    arch: str = "resnet50"
    workers: int = 8
    batch_size: int = 128
    cos_steps: int = 190
    lr: float = 0.03
    schedule: tuple = (120, 160)
    momentum: float = 0.9
    weight_decay: float = 1e-4

    # moco specific configs:
    moco_dim: int = 128
    moco_k: int = 16_384  # queue size; number of negative keys (default: 65536)
    moco_m: float = 0.999  # moco momentum of updating key encoder (default: 0.999)
    moco_t: float = 0.2  # softmax temperature (default: 0.07)"

    # options for moco v2
    mlp: bool = True  # use mlp head
    aug_plus: bool = True  # use moco v2 data augmentation
    cos: bool = True  # use cosine lr schedule

    # region MOCHI PARAMETERS
    mochi: bool = True  # use mochi

    mochi_N: int = 1024  # hard negative pool size
    mochi_s: int = 1024  # number of harder negative btw 2 pairs of hard negatives
    mochi_s_prime: int = 128  # number of harder negative btw a negative and positive
    mochi_tau: float = SI("${pretraining_params.moco_t}")
    mochi_warmup: int = (
        10  # learning rate warmup epochs + do not synthesize hard negatives
    )
