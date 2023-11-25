from typing import Any

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from jaxtyping import Float
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    LinearLR,
    MultiStepLR,
    SequentialLR,
)

import moco
import moco.builder
import moco.loader
from conf.pretrain_params import PretrainParams
from main_moco import AverageMeter, accuracy

model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)


class Metrics(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.top1 = AverageMeter("Acc@1", ":6.2f")
        self.top5 = AverageMeter("Acc@5", ":6.2f")

    def update(self, output, target, batch_size):
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        self.top1.update(acc1[0], batch_size)
        self.top5.update(acc5[0], batch_size)

    def reset(self):
        self.top1.reset()
        self.top5.reset()


class PretrainingLightningModule(pl.LightningModule):
    def __init__(
        self,
        args: PretrainParams,
    ):
        super().__init__()
        self.args = args

        # create model
        print("=> creating model '{}'".format(args.arch))
        model = moco.builder.MoCo(
            models.__dict__[args.arch],
            args.moco_dim,
            args.moco_k,
            args.moco_m,
            args.moco_t,
            args.mlp,
        )
        self.model = model
        model.args = args

        criterion = nn.CrossEntropyLoss()
        self.criterion = criterion

        self.train_metrics = Metrics()
        self.valid_metrics = Metrics()
        self.test_metrics = Metrics()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )

        warmup_scheduler = LinearLR(
            optimizer=optimizer,
            total_iters=self.args.mochi_warmup,
            verbose=True,
        )

        """Decay the learning rate based on schedule"""
        if self.args.cos:  # cosine lr schedule
            scheduler = CosineAnnealingLR(
                optimizer=optimizer,
                T_max=self.args.cos_steps,
                eta_min=0,
                verbose=True,
            )
        else:  # stepwise lr schedule
            scheduler = MultiStepLR(
                optimizer=optimizer,
                milestones=self.args.schedule,
                gamma=0.1,
                verbose=True,
            )

        sequentialLR = SequentialLR(
            optimizer=optimizer,
            schedulers=[warmup_scheduler, scheduler],
            milestones=[warmup_scheduler.total_iters + 1],
            verbose=True,
        )

        return [optimizer], [sequentialLR]

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log_g("train", "lr", self.trainer.optimizers[0].param_groups[0]["lr"])
        return loss

    def validation_step(self, batch, batch_idx):
        self._step(batch)

    def test_step(self, batch, batch_idx):
        self._step(batch)

    def _step(self, images: list[torch.Tensor]) -> torch.Tensor:
        # compute output
        images, _ = images
        output, target = self.model(
            im_q=images[0], im_k=images[1], is_mochi=self.args.mochi
        )
        loss = self.criterion(output, target)
        self.get_metric_object().update(self._remove_synthetic_for_accuracy(output).clone().detach(), target.clone().detach(), batch_size=images[0].size(0))

        self.log_g(self.get_stage(), "loss", loss.item(), prog_bar=True)

        return loss

    def _remove_synthetic_for_accuracy(self, output: Float[torch.Tensor, "b one_k_s_sprime"]) -> Float[torch.Tensor, "b one_k"]:
        used = 1 + self.args.moco_k
        return output[:, :used]

    def on_train_epoch_end(self) -> None:
        self.log_g("train", "acc1", self.train_metrics.top1.avg, prog_bar=True)
        self.log_g("train", "acc5", self.train_metrics.top5.avg, prog_bar=True)
        self.train_metrics.reset()

    def on_validation_epoch_end(self) -> None:
        self.log_g("valid", "acc1", self.valid_metrics.top1.avg, prog_bar=True)
        self.log_g("valid", "acc5", self.valid_metrics.top5.avg, prog_bar=True)
        self.valid_metrics.reset()

    def on_test_epoch_end(self) -> None:
        self.log_g("test", "acc1", self.test_metrics.top1.avg, prog_bar=True)
        self.log_g("test", "acc5", self.test_metrics.top5.avg, prog_bar=True)
        self.test_metrics.reset()

    def get_metric_object(self):
        if self.trainer.training:
            return self.train_metrics
        elif self.trainer.validating or self.trainer.sanity_checking:
            return self.valid_metrics
        elif self.trainer.testing or self.trainer.predicting:
            return self.test_metrics
        else:
            raise Exception(f"Stage not supported.")

    def get_stage(self) -> str:
        if self.trainer.training:
            return "train"
        elif self.trainer.validating or self.trainer.sanity_checking:
            return "valid"
        elif self.trainer.testing or self.trainer.predicting:
            return "test"
        else:
            raise Exception(f"Stage not supported.")

    def log_g(self, train_stage: str, logged: str, value: Any, **kwargs):
        self.log(
            f"{train_stage}/{logged}",
            value,
            **kwargs,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
        )
