import os
import random
import warnings
from dataclasses import asdict
from pprint import pprint

import hydra
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from omegaconf import OmegaConf

import moco.builder
import moco.loader
import wandb
from conf.checkpoint_params import CheckpointsCallbacks, getModelCheckpoint
from conf.main_config import GlobalConfiguration
from conf.trainer_params import get_trainer
from conf.wandb_params import get_wandb_logger
from LightningModulePretraining import PretrainingLightningModule


def get_train_dataloader(args):
    # Data loading code
    traindir = os.path.join(args.data, "train")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    if args.aug_plus:
        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomApply(
                [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8  # not strengthened
            ),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([moco.loader.GaussianBlur([0.1, 2.0])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    else:
        # MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
        augmentation = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
            transforms.RandomGrayscale(p=0.2),
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]

    train_dataset = datasets.ImageFolder(
        traindir, moco.loader.TwoCropsTransform(transforms.Compose(augmentation))
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        drop_last=True,
    )

    return train_loader


@hydra.main(version_base=None, config_name="globalConfiguration")
def main(_cfg: GlobalConfiguration):
    if _cfg.yaml_conf is not None:
        _cfg = OmegaConf.merge(
            _cfg, OmegaConf.load(_cfg.yaml_conf)
        )  # command line configuration + yaml configuration
    _cfg = OmegaConf.merge(
        _cfg, {key: val for key, val in OmegaConf.from_cli().items() if "/" not in key}
    )  # command line configuration + yaml configuration + command line configuration

    cfg: GlobalConfiguration = OmegaConf.to_object(_cfg)

    args = cfg.pretraining_params

    print(OmegaConf.to_yaml(_cfg))
    pprint(vars(args))

    if cfg.seed is not None:
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        cudnn.deterministic = True
        warnings.warn(
            "You have chosen to seed training. "
            "This will turn on the CUDNN deterministic setting, "
            "which can slow down your training considerably! "
            "You may see unexpected behavior when restarting "
            "from checkpoints."
        )
    cudnn.benchmark = True

    train_loader = get_train_dataloader(args)

    run_wandb = get_wandb_logger(
        params=cfg.wandb_params,
        global_dict=vars(args) | asdict(cfg),
        additional_conf=None,
    )

    callbacks = []
    modelCheckpoint: CheckpointsCallbacks = getModelCheckpoint(cfg.checkpoint_params)
    callbacks += [modelCheckpoint.on_monitor, modelCheckpoint.on_duration]

    trainer = get_trainer(cfg.trainer_params, callbacks, run_wandb)

    model = PretrainingLightningModule(args=args)

    if cfg.checkpoint_params.retrain_from_checkpoint == 'load_weights':
        print(f"load weights from {cfg.checkpoint_params.retrain_saved_path}")
        missing_keys, unexpected_keys = model.load_state_dict(
            state_dict=torch.load(cfg.checkpoint_params.retrain_saved_path)['state_dict'],
        )
        print(f"missing_keys: {missing_keys}")
        print(f"unexpected_keys: {unexpected_keys}")

    # Train
    if cfg.trainer_params.skip_training:
        print("skip training")
    else:
        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=None,
            ckpt_path=cfg.checkpoint_params.retrain_saved_path if cfg.checkpoint_params.retrain_from_checkpoint == 'load_train' else None,
        )
        print("end fitting")

    if cfg.trainer_params.exit_after_training:
        print("exit after training")
        print(f"<TERMINATE WANDB>")
        wandb.finish()
        print(f"<WANDB TERMINATED>")
        return


if __name__ == "__main__":
    main()
