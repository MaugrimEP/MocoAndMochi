from dataclasses import dataclass


@dataclass
class SlurmParams:
    job_id: int = -1
    working_directory: str = "None"
    slurm_user: str = "user_nf"
