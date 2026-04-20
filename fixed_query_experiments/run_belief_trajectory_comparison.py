import logging

import hydra
from omegaconf import OmegaConf

from fixed_query_experiments.belief_trajectory_comparison import run
import utils


@hydra.main(version_base=None, config_path="../conf", config_name="fixed-query-belief-comparison")
def main(cfg):
    OmegaConf.resolve(cfg)
    logging.info(f"Hydra version: {hydra.__version__}")
    logging.info(OmegaConf.to_yaml(cfg))
    run(cfg)


if __name__ == "__main__":
    OmegaConf.register_new_resolver("githash", utils.githash)
    main()
