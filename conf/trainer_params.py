from dataclasses import dataclass, field
from typing import List, Optional, Union

import pytorch_lightning as pl


@dataclass
class TrainerParams:
    max_time: Optional[str] = None  # None  # DD:HH:MM:SS (days, hours, minutes seconds)

    accelerator: str = "gpu"  # [ gpu | cpu ]

    strategy: Optional[str] = "auto"
    devices: Union[str, int] = 1
    precision: Union[str, int] = 32

    num_nodes: int = 1

    plugins: List[str] = field(default_factory=lambda: [])

    check_val_every_n_epoch: Optional[
        int
    ] = 1  # perform valid every N epoch, or set to None and
    # validation will be checked every val_check_interval batches
    log_every_n_steps: int = 1  # How often to log within steps. Default: 50.
    accumulate_grad_batches: Optional[int] = 1

    max_epochs: int = 200
    max_steps: int = -1

    # parameters not related to pl trainer
    benchmark: Optional[bool] = True
    cudnn_benchmark: Optional[bool] = True

    num_sanity_val_steps: int = 0

    fast_dev_run: Union[
        bool, int
    ] = False  # Runs n if set to n (int) else 1 if set to True batch(es) of train, val and test to find any bugs (ie: a sort of unit test). Default: False.
    limit_train_batches: Union[
        int, float, None
    ] = 1.0  # How much of training dataset to check (float = fraction, int = num_batches). Default: 1.0.
    limit_val_batches: Union[
        int, float, None
    ] = 1.0  # How much of validation dataset to check (float = fraction, int = num_batches). Default: 1.0.
    limit_test_batches: Union[
        int, float, None
    ] = 1.0  # How much of test dataset to check (float = fraction, int = num_batches). Default: 1.0.
    overfit_batches: Union[
        int, float
    ] = 0.0  # Overfit a fraction of training/validation data (float) or a set number of batches (int). Default: 0.0.

    gradient_clip_val: Union[int, float, None] = None
    gradient_clip_algorithm: Optional[str] = None

    ####
    skip_training: bool = False
    exit_after_training: bool = False


def get_trainer(trainer_params, callbacks: List, logger):
    trainer = pl.Trainer(
        max_epochs=trainer_params.max_epochs,
        max_steps=trainer_params.max_steps,
        max_time=trainer_params.max_time,
        accelerator=trainer_params.accelerator,
        devices=trainer_params.devices,
        check_val_every_n_epoch=trainer_params.check_val_every_n_epoch,
        callbacks=callbacks,
        logger=logger,
        enable_checkpointing=True,
        log_every_n_steps=trainer_params.log_every_n_steps,
        accumulate_grad_batches=trainer_params.accumulate_grad_batches,
        ###
        strategy=trainer_params.strategy,
        precision=trainer_params.precision,
        num_nodes=trainer_params.num_nodes,
        plugins=list(trainer_params.plugins),
        benchmark=trainer_params.benchmark,
        num_sanity_val_steps=trainer_params.num_sanity_val_steps,
        fast_dev_run=trainer_params.fast_dev_run,
        limit_train_batches=trainer_params.limit_train_batches,
        limit_val_batches=trainer_params.limit_val_batches,
        limit_test_batches=trainer_params.limit_test_batches,
        overfit_batches=trainer_params.overfit_batches,
        gradient_clip_val=trainer_params.gradient_clip_val,
        gradient_clip_algorithm=trainer_params.gradient_clip_algorithm,
    )
    return trainer
