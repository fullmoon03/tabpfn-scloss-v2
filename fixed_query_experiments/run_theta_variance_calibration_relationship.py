import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from fixed_query_experiments.theta_variance_calibration_relationship import run
import utils


@hydra.main(
    version_base=None,
    config_path="../conf",
    config_name="theta-variance-calibration-relationship",
)
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    logging.info(f"Hydra version: {hydra.__version__}")
    logging.info(OmegaConf.to_yaml(cfg))
    run(cfg)


if __name__ == "__main__":
    OmegaConf.register_new_resolver("githash", utils.githash)
    main()
