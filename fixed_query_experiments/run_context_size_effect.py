import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from fixed_query_experiments.context_size_effect import run
import utils


@hydra.main(version_base=None, config_path="../conf", config_name="context-size-effect")
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
