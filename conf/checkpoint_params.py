from dataclasses import dataclass
from typing import Optional

from omegaconf import SI
from pytorch_lightning.callbacks import ModelCheckpoint


@dataclass
class CheckpointParams:
    loading_for_test_mode: str = "none"  # [ none | monitor | last ]

    # generals params
    dirpath: str = "_model_save/"
    auto_insert_metric_name: bool = False
    save_weights_only: bool = False
    verbose: bool = True
    save_on_train_epoch_end: Optional[
        bool
    ] = True  # without this, does not save at all :think:
    save_last: Optional[
        bool
    ] = True  # create last.ckpt (for each checkpoint: cp checkpoint.ckpt last.ckpt)

    #################################
    #################################

    # region ON MONITOR
    on_monitor__filename: Optional[str] = "{epoch:04d}_best_model_{valid/acc1:.5f}"
    monitor: Optional[str] = "valid/acc1"  # if None save to the last epoch
    mode: str = "max"
    on_monitor__every_n_epochs: Optional[int] = 1
    on_monitor__save_top_k: int = 1
    # endregion

    #################################

    # region ON DURATION
    on_duration__filename: Optional[str] = "{epoch:04}_duration_model"
    on_duration__save_top_k: int = 1
    on_duration__every_n_epochs: Optional[int] = 1
    # endregion

    # region CONTINUE TRAINING FROM A CKTP
    retrain_saved_path: Optional[str] = None
    retrain_from_checkpoint: str = "none"  # [ none | load_train | load_weights ]
    """
    load_train: restart pytorch lightning training from a PL ckpt
    load_weights: load weights from a ckpt, and do not load PL
    """


@dataclass
class CheckpointsCallbacks:
    on_monitor: Optional[ModelCheckpoint]
    on_duration: Optional[ModelCheckpoint]


def getModelCheckpoint(params: CheckpointParams) -> CheckpointsCallbacks:
    on_monitor = ModelCheckpoint(
        dirpath=params.dirpath,
        filename=params.on_monitor__filename,
        monitor=params.monitor,
        verbose=params.verbose,
        save_last=params.save_last,
        save_top_k=params.on_monitor__save_top_k,
        mode=params.mode,
        auto_insert_metric_name=params.auto_insert_metric_name,
        save_weights_only=params.save_weights_only,
        every_n_epochs=params.on_monitor__every_n_epochs,
        save_on_train_epoch_end=params.save_on_train_epoch_end,
    )
    on_duration = ModelCheckpoint(
        dirpath=params.dirpath,
        filename=params.on_duration__filename,
        auto_insert_metric_name=params.auto_insert_metric_name,
        save_weights_only=params.save_weights_only,
        monitor="epoch",
        mode="max",
        save_top_k=params.on_duration__save_top_k,
        verbose=params.verbose,
        save_last=params.save_last,
        every_n_epochs=params.on_duration__every_n_epochs,
        save_on_train_epoch_end=params.save_on_train_epoch_end,
    )

    return CheckpointsCallbacks(on_monitor=on_monitor, on_duration=on_duration)
