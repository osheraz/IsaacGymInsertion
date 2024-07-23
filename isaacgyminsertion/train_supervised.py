import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from termcolor import cprint
from isaacgyminsertion.utils.utils import set_np_formatting, set_seed

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


@hydra.main(config_name="config", config_path="./cfg")
def run(cfg: DictConfig):  # , config_path: Optional[str] = None

    if cfg.checkpoint:
        cfg.checkpoint = to_absolute_path(cfg.checkpoint)

    # set numpy formatting for printing only
    set_np_formatting()


    if cfg.train_diffusion:
        from algo.models.diffusion.train_diffusion import Runner

        # perform train
        runner = Runner(cfg)

        exit()

    if cfg.train_tactile:
        from algo.models.transformer.tactile_runner import Runner

        # perform train
        runner = Runner(cfg, agent=None)
        runner.run()

        exit()

    # for training the model with offline data only
    if cfg.offline_training:
        from algo.models.transformer.runner import Runner

        # perform train
        runner = Runner(cfg, agent=None)
        runner.run()

        exit()

    cprint("Start Building the Environment", "green", attrs=["bold"])



if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument(
    #     "config_path", type=argparse.FileType("r"), help="Path to hydra config."
    # )
    # args = parser.parse_args()

    run() # None, args.config_path
