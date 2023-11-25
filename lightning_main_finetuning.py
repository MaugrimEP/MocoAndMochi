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

import wandb
from conf.checkpoint_params import CheckpointsCallbacks, getModelCheckpoint
from conf.main_config import GlobalConfiguration
from conf.trainer_params import get_trainer
from conf.wandb_params import get_wandb_logger
from LightningModuleFinetuning import FineTuningModule


def get_dataloaders(args):
    # Data loading code
    traindir = os.path.join(args.data, "train")
    valdir = os.path.join(args.data, "val")
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]
        ),
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(
            valdir,
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        ),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    return train_loader, val_loader


@hydra.main(version_base=None, config_name="globalConfiguration")
def main(_cfg: GlobalConfiguration):
    # region fetch configuration
    if _cfg.yaml_conf is not None:
        _cfg = OmegaConf.merge(
            _cfg, OmegaConf.load(_cfg.yaml_conf)
        )  # command line configuration + yaml configuration
    _cfg = OmegaConf.merge(
        _cfg, {key: val for key, val in OmegaConf.from_cli().items() if "/" not in key}
    )  # command line configuration + yaml configuration + command line configuration

    cfg: GlobalConfiguration = OmegaConf.to_object(_cfg)

    args = cfg.lincls_params

    print(OmegaConf.to_yaml(_cfg))
    pprint(vars(args))
    # endregion

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

    train_dataloader, val_loader = get_dataloaders(args)

    run_wandb = get_wandb_logger(
        params=cfg.wandb_params,
        global_dict=vars(args) | asdict(cfg),
        additional_conf=None,
    )

    callbacks = []
    modelCheckpoint: CheckpointsCallbacks = getModelCheckpoint(cfg.checkpoint_params)
    callbacks += [modelCheckpoint.on_monitor, modelCheckpoint.on_duration]

    trainer = get_trainer(cfg.trainer_params, callbacks, run_wandb)

    model = FineTuningModule(args=args)
    if cfg.checkpoint_params.retrain_from_checkpoint == 'load_weights':
        print(f"load weights from {cfg.checkpoint_params.retrain_saved_path}")
        missing_keys, unexpected_keys = model.load_state_dict(
            state_dict=torch.load(cfg.checkpoint_params.retrain_saved_path)['state_dict'],
        )
        print(f"missing_keys: {missing_keys}")
        print(f"unexpected_keys: {unexpected_keys}")
    """
    Switch to eval mode:
    Under the protocol of linear classification on frozen features/models,
    it is not legitimate to change any part of the pre-trained model.
    BatchNorm in train mode may revise running mean/std (even if it receives
    no gradient), which are part of the model parameters too.
    """
    model.model.eval()  # TODO check that is stay during training

    # Train
    if cfg.trainer_params.skip_training:
        print("skip training")
    else:
        trainer.fit(
            model,
            train_dataloaders=train_dataloader,
            val_dataloaders=val_loader,
            ckpt_path=cfg.checkpoint_params.retrain_saved_path if cfg.checkpoint_params.retrain_from_checkpoint == 'load_train' else None,
        )
        print("end fitting")

    if cfg.trainer_params.exit_after_training:
        print("exit after training")
        print(f"<TERMINATE WANDB>")
        wandb.finish()
        print(f"<WANDB TERMINATED>")
        return

    trainer.validate(
        model,
        dataloaders=val_loader,
    )


if __name__ == "__main__":
    main()
