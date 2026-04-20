import logging
import os
import pickle
import re
import subprocess
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import chex
import jax
import jax.numpy as jnp


def suppress_noisy_third_party_logs():
    """Hide known non-actionable third-party warnings from experiment logs."""

    class _NoisyThirdPartyFilter(logging.Filter):
        def filter(self, record):
            if record.name.startswith(("posthog", "analytics", "analytics-python")):
                return False
            if "analytics-python queue is full" in record.getMessage():
                return False
            return True

    for logger_name in ("posthog", "analytics", "analytics-python"):
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.CRITICAL + 1)
        logger.disabled = True
        logger.propagate = False

    root_logger = logging.getLogger()
    root_logger.addFilter(_NoisyThirdPartyFilter())
    for handler in root_logger.handlers:
        handler.addFilter(_NoisyThirdPartyFilter())


def get_n_data(data):
    # return the number of samples in 'data' which is a dictionary of arrays
    # with the same leading dimension

    leading_dims = [x.shape[0] for x in data.values()]
    assert all([x == leading_dims[0] for x in leading_dims])
    return leading_dims[0]


def get_tree_lead_dim(tree):
    # Return the leading dimensions of a PyTree, assuming that all leaves have
    # the same leading dimension

    leaves = jax.tree.leaves(tree)
    chex.assert_equal_shape_prefix(leaves, 1)
    return leaves[0].shape[0]


def tree_shape(tree):
    return jax.tree.map(lambda x: jnp.shape(x), tree)


def githash():
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode("ascii")
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"


def kst_hhmm():
    return datetime.now(ZoneInfo("Asia/Seoul")).strftime("%H%M")


def tuned_path_suffix(path):
    if path is None:
        return "nosuffix"

    candidate = Path(str(path))
    for path_part in (candidate, *candidate.parents):
        match = re.search(r"_(\d{4})$", path_part.name)
        if match:
            return match.group(1)
    return "nosuffix"


def write_to_local(path, obj, verbose=False):
    # write to local
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    if verbose:
        logging.info(f"Write to local:/{path}")


def write_to(path, obj, verbose=False):
    write_to_local(path, obj, verbose=verbose)


def read_from_gs(bucket_name, path):
    # bucket: bucket name
    # path: path to the file
    # obj: the object to save

    bucket = get_bucket(bucket_name)
    blob = bucket.blob(path)
    with blob.open("rb") as f:
        obj = pickle.load(f)
    return obj


def read_from_local(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
    return obj


def read_from(path):
    if path.startswith("gs://"):
        # read from gcs
        bucket_name, path = path.split("/", 3)[2:]
        return read_from_gs(bucket_name, path)
    else:
        # read from local
        return read_from_local(path)


def print_dgp(d):
    return " ".join([f"{k}={v}" for k, v in d.items()])


def get_data_size(path):
    match = re.search(r"data=([^ ]+)", path)
    if match:
        return match.group(1)
    return None


def get_resample_x(path):
    match = re.search(r"resample_x=([^ ]+)", path)
    if match:
        return match.group(1)
    return None


def get_seed(path):
    match = re.search(r"seed=([^ ]+)", path)
    if match:
        return int(match.group(1))
    return None


def get_dim_x(path):
    match = re.search(r"dim_x=([^ ]+)", path)
    if match:
        return int(match.group(1))
    return None


def get_date_part(path):
    match = re.search(r"outputs/([^/]+)/", path)
    if match:
        return match.group(1)
    return None


def format_decimal(x, decimals=2):
    if x:
        return f"{x:.{decimals}f}"
    return None


def get_data_name(path):
    match = re.search(r"name=([^ ]+)", path)
    if match:
        if match.group(1) == "classification-fixed":
            return "classification-standard"
        elif match.group(1) == "classification-fixed-gmm":
            match2 = re.search(r"a=([^ ]+)", path)
            return f"classification-gmm-{match2.group(1)}"
        elif match.group(1) == "regression-fixed-dependent":
            match2 = re.search(r"s_small=([^ ]+)", path)
            match3 = re.search(r"s_mod=([^ ]+)", path)
            return f"regression-dependent-{match2.group(1)}-{match3.group(1)}"
        elif match.group(1) == "regression-fixed":
            return "regression-standard"
        elif match.group(1) == "regression-fixed-non-normal":
            match2 = re.search(r"df=([^ ]+)", path)
            return f"regression-t-{match2.group(1)}"
        return match.group(1)
    return None
