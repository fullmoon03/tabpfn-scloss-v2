import logging

import hydra
from omegaconf import OmegaConf

from fixed_query_experiments.baseline_tuned_calibration import run
import utils


@hydra.main(version_base=None, config_path="../conf", config_name="baseline-tuned-calibration")
def main(cfg):
    OmegaConf.resolve(cfg)
    logging.info(f"Hydra version: {hydra.__version__}")
    logging.info(OmegaConf.to_yaml(cfg))
    run(cfg)


if __name__ == "__main__":
    OmegaConf.register_new_resolver("githash", utils.githash)
    main()
