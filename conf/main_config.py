from dataclasses import dataclass, field
from typing import Any, List, Optional

from hydra.core.config_store import ConfigStore

from conf.checkpoint_params import CheckpointParams
from conf.lincls_params import LinclsParams
from conf.pretrain_params import PretrainParams
from conf.slurm_params import SlurmParams
from conf.trainer_params import TrainerParams
from conf.wandb_params import WandbParams


@dataclass
class GlobalConfiguration:
    # region default values
    defaults: List[Any] = field(
        default_factory=lambda: [
            "_self_",
        ]
    )
    # endregion

    seed: Optional[int] = None
    yaml_conf: Optional[str] = None

    trainer_params: TrainerParams = TrainerParams()
    wandb_params: WandbParams = WandbParams()
    checkpoint_params: CheckpointParams = CheckpointParams()
    slurm_params: SlurmParams = SlurmParams()

    lincls_params: LinclsParams = LinclsParams()
    pretraining_params: PretrainParams = PretrainParams()


# region register config
cs = ConfigStore.instance()
cs.store(name="globalConfiguration", node=GlobalConfiguration)
# endregion
