from dataclasses import dataclass

import torchvision.models as models

model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)


@dataclass
class LinclsParams:
    data: str = r"/home/2022022/PARTAGE/Imagenet-100"
    r"""
    C:\Users\tmayet\MobaXterm\home\Imagenet-100
    /home/2022022/PARTAGE/Imagenet-100
    """
    arch: str = "resnet50"
    workers: int = 8
    batch_size: int = 128
    lr: float = 10.0
    schedule: tuple = (30, 40, 50)
    momentum: float = 0.9
    weight_decay: float = 0.0
    evaluate: bool = False

    pretrained: str = ""  # path to moco pretrained checkpoint

    cos: bool = False
    fine_tuning_warmup_epochs: int = 0
