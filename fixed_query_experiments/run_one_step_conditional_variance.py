import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from fixed_query_experiments.one_step_conditional_variance import run
import utils


@hydra.main(version_base=None, config_path="../conf", config_name="conditional-theta-variance")
def main(cfg: DictConfig) -> None:
    OmegaConf.resolve(cfg)
    utils.suppress_noisy_third_party_logs()
    logging.info(f"Hydra version: {hydra.__version__}")
    logging.info(OmegaConf.to_yaml(cfg))
    run(cfg)


if __name__ == "__main__":
    utils.suppress_noisy_third_party_logs()
    OmegaConf.register_new_resolver("githash", utils.githash)
    main()
