import os
from typing import Any
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.models as models
from torch.optim.lr_scheduler import MultiStepLR

from conf.lincls_params import LinclsParams
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


class FineTuningModule(pl.LightningModule):
    def __init__(
        self,
        args: LinclsParams,
    ):
        super().__init__()
        self.args = args

        # create model
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()

        # freeze all layers but the last fc
        for name, param in model.named_parameters():
            if name not in ["fc.weight", "fc.bias"]:
                param.requires_grad = False
        # init the fc layer
        model.fc = nn.Linear(2048, 100)
        # model.fc.weight.data.normal_(mean=0.0, std=0.01)
        # model.fc.bias.data.zero_()
        self.model = model

        # load from pre-trained, before DistributedDataParallel constructor
        if args.pretrained:
            if os.path.isfile(args.pretrained):
                print("=> loading checkpoint '{}'".format(args.pretrained))
                checkpoint = torch.load(args.pretrained, map_location="cpu")

                # rename moco pre-trained keys
                state_dict = checkpoint["state_dict"]
                prefix = list(state_dict.keys())[0].split('.')[0]
                print(f"used prefix for loading {prefix=} from {list(state_dict.keys())[0]=}")
                for k in list(state_dict.keys()):
                    # retain only encoder_q up to before the embedding layer
                    if k.startswith(f"{prefix}.encoder_q") and not k.startswith(
                        f"{prefix}.encoder_q.fc"
                    ):
                        # remove prefix
                        state_dict[k[len(f"{prefix}.encoder_q.") :]] = state_dict[k]
                    # delete renamed or unused k
                    del state_dict[k]

                args.start_epoch = 0
                msg = model.load_state_dict(state_dict, strict=False)
                assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}

                print("=> loaded pre-trained model '{}'".format(args.pretrained))
            else:
                print("=> no checkpoint found at '{}'".format(args.pretrained))

        # define loss function (criterion) and optimizer
        self.criterion = nn.CrossEntropyLoss()

        self.train_metrics = Metrics()
        self.valid_metrics = Metrics()
        self.test_metrics = Metrics()

    def configure_optimizers(self):
        # optimize only the linear classifier
        parameters = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        assert len(parameters) == 2  # fc.weight, fc.bias
        print(f"{len(parameters)=}")
        optimizer = torch.optim.SGD(
            parameters,
            self.args.lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay,
        )

        scheduler = MultiStepLR(
            optimizer=optimizer,
            milestones=self.args.schedule,
            gamma=0.1,
            verbose=True,
        )

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log_g("train", "lr", self.trainer.optimizers[0].param_groups[0]["lr"])
        return loss

    def validation_step(self, batch, batch_idx):
        self._step(batch)

    def test_step(self, batch, batch_idx):
        self._step(batch)

    def _step(self, batch: list[torch.Tensor]) -> torch.Tensor:
        # compute output
        images, target = batch

        # compute output
        output = self.model(images)
        loss = self.criterion(output, target)

        self.get_metric_object().update(output, target, batch_size=images[0].size(0))
        self.log_g(self.get_stage(), "loss", loss.item(), prog_bar=True)

        return loss

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
