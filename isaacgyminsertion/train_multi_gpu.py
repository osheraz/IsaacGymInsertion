import pathlib
import sys
import tempfile
import traceback

import hydra
import submitit
from omegaconf import DictConfig, OmegaConf
from hydra.core.hydra_config import HydraConfig


def launch_submitit(config):
    launcher_cfg = HydraConfig.get().launcher
    assert launcher_cfg.tasks_per_node == 1 and launcher_cfg.nodes == 1
    with tempfile.NamedTemporaryFile() as fp:
        OmegaConf.save(config=config, f=fp.name)
        if config.train.ppo.multi_gpu:
            exec_cmd = [
                "torchrun",
                "--standalone",
                "--nnodes=1",
                f"--nproc_per_node={launcher_cfg.gpus_per_node}",
            ]
        else:
            exec_cmd = ["python"]
        function = submitit.helpers.CommandFunction(
            exec_cmd + ["train.py", f"{fp.name}"]
        )
        executor = submitit.AutoExecutor(
            folder=pathlib.Path(launcher_cfg.submitit_folder).parent / "train_logs",
            cluster="local",
        )
        executor.update_parameters(
            timeout_min=launcher_cfg.timeout_min,
            cpus_per_task=launcher_cfg.cpus_per_task,
            gpus_per_node=launcher_cfg.gpus_per_node,
            mem_gb=launcher_cfg.mem_gb,
        )
        job = executor.submit(function)
        print(job.result())


@hydra.main(config_name="config", config_path="./cfg")
def main(config: DictConfig):
    try:
        launcher_cfg = HydraConfig.get().launcher
        if "submitit" in launcher_cfg._target_:
            launch_submitit(config)
        else:
            from train import run

            run(config)

    except Exception:
        traceback.print_exc(file=sys.stderr)
        raise


if __name__ == "__main__":
    main()