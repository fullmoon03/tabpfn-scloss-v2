import logging
import os
from abc import abstractmethod

import equinox as eqx
import jax
import numpy as np
import openml
import pandas as pd
from jaxtyping import Array, ArrayLike, PRNGKeyArray
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import KBinsDiscretizer
from ucimlrepo import fetch_ucirepo

from synthetic_dgp import (
    ClassificationFixedGenerator,
    ClassificationFixedGMMLinkGenerator,
    ClassificationLinearGenerator,
    ClassificationPriorGenerator,
    DependentErrorWMGenerator,
    LinearRegressionWMGenerator,
    NonNormalErrorWMGenerator,
    RegressionFixedDependentErrorGenerator,
    RegressionFixedGenerator,
    RegressionFixedNonNormalErrorGenerator,
    RegressionPriorGenerator,
    SCMNNClassificationGenerator,
    SCMDagClassificationGenerator,
)
import utils

openml.config.set_root_cache_directory(os.path.join(os.getcwd(), "datasets/openml"))


class DGP(eqx.Module):
    input_key: PRNGKeyArray
    train_data: dict[str, np.ndarray]

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def get_x_data(self, key: PRNGKeyArray, n: int) -> Array:
        """
        This will be used as part of the forward recursion and could be inside a
        JAX transformed function.  It should only contain JAX
        transformation-compatible operations.
        """
        pass

def multidim_stratified_split(
    key: PRNGKeyArray,
    X: ArrayLike,
    y: ArrayLike,
    is_X_categorical: list[bool],
    is_y_categorical: bool,
    train_size: int,
    n_bins: int,
    continuous_threshold: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform a stratified split of the data into training and test sets.
    Each dimension of x and y are stratified. For continuous variables, they are
    binned into `n_bins` bins.

    Args:
        key: JAX random key for reproducibility.
        X: Features data as a 2D array.
        y: Target data as a 1D array.
        is_X_categorical: List indicating whether each feature in X is categorical.
        is_y_categorical: Whether the target variable y is categorical.
        train_size: Number of samples to include in the training set.
        n_bins: Number of bins to use for continuous variables.

    Returns:
        X_train: Training features.
        X_test: Test features.
        y_train: Training target.
        y_test: Test target.
    """

    seed = jax.random.randint(key, shape=(), minval=0, maxval=42949672).item()

    # Convert to DataFrame for easier handling
    df = pd.DataFrame(np.hstack([y[:, None], X]))
    is_categorical = [is_y_categorical] + is_X_categorical

    # for splitting purpose, any continuous variable with less than 4 * n_bins unique values is treated as categorical
    strata_keys = []
    continuous_indices = []
    categorical_indices = []
    continuous_threshold = continuous_threshold or 4 * n_bins
    for i in range(df.shape[1]):
        if is_categorical[i]:
            categorical_indices.append(i)
        else:
            unique_values = df.iloc[:, i].nunique()
            if unique_values > continuous_threshold:
                # Treat as continuous
                continuous_indices.append(i)
            else:
                # Treat as categorical
                categorical_indices.append(i)

    # Bin continuous variables
    if continuous_indices:
        if n_bins == 1:
            continuous_binned = np.zeros((df.shape[0], len(continuous_indices)))
        else:
            kbd = KBinsDiscretizer(n_bins=n_bins, encode="ordinal", strategy="quantile")
            continuous_binned = kbd.fit_transform(df.iloc[:, continuous_indices])

        logging.debug("Number of data in each bin (for x treated as continuous):")
        for i, col_idx in enumerate(continuous_indices):
            strata_keys.append(continuous_binned[:, i].astype(int).astype(str))
            logging.debug(
                f"x[{col_idx}]: {np.bincount(continuous_binned[:, i].astype(int), minlength=n_bins)}"
            )

    if categorical_indices:
        logging.debug("Number of data in each bin (for x treated as categorical):")
        # Add categorical variables directly
        for col_idx in categorical_indices:
            strata_keys.append(df.iloc[:, col_idx].astype(str))
            logging.debug(
                f"x[{col_idx}]: {df.iloc[:, col_idx].value_counts().to_numpy()}"
            )

    # Combine all features into single stratification key
    strata = pd.Series(["_".join(row) for row in zip(*strata_keys)])

    # Remove strata with too few samples
    strata_counts = strata.value_counts()
    valid_strata = strata_counts[strata_counts >= 2].index
    valid_mask = strata.isin(valid_strata)

    # Split only valid samples
    X_valid = X[valid_mask]
    y_valid = y[valid_mask]
    strata_valid = strata[valid_mask]

    logging.debug(f"Strata counts: {strata_counts.to_numpy()}")
    logging.debug(f"Valid mask proportion: {sum(valid_mask) / valid_mask.size}")
    train_test_datasets = train_test_split(
        X_valid,
        y_valid,
        train_size=train_size,
        stratify=strata_valid,
        random_state=seed,
    )
    X_train, X_test, y_train, y_test = jax.tree.map(np.asarray, train_test_datasets)

    return X_train, X_test, y_train, y_test


class DGPReal(DGP):
    """
    These are the necessary variables for all read data DGPs.  It defines how
    the population is defined, how new x is drawn from the population, and how
    the data are split.

    By design the datasets are numpy arrays instead of JAX arrays, because the
    downloaded array might contain strings or mixed types columns.
    """

    full_data: dict[str, np.ndarray]
    test_data: dict[str, np.ndarray]
    categorical_x: list[bool]
    strata_bins: int

    def get_x_data(self, key: PRNGKeyArray, n: int) -> Array:
        # Sample x from self.full_data with replacement
        if self.full_data is None:
            raise ValueError("Full data is not available. Cannot sample x.")
        if "x" not in self.full_data:
            raise ValueError("Full data does not contain 'x' key.")
        key, subkey = jax.random.split(key)
        indices = jax.random.choice(subkey, self.full_data["x"].shape[0], shape=(n,))
        return self.full_data["x"][indices]

    def split_data(
        self,
        key: PRNGKeyArray,
        n: int,
        is_y_categorical: bool,
        continuous_threshold: int | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        """
        Perform a stratified split of the full data into training and test sets.
        Each dim of x and y are stratified.  For continuous variables, they are
        binned into `n_bins` bins.

        Args: key: JAX random key for reproducibility.  n: Number of samples to
        include in the training set.  is_y_categorical: Whether the target
        variable is categorical.
        """

        x_train, x_test, y_train, y_test = multidim_stratified_split(
            key,
            self.full_data["x"],
            self.full_data["y"],
            self.categorical_x,
            is_y_categorical=is_y_categorical,
            train_size=n,
            n_bins=self.strata_bins,
            continuous_threshold=continuous_threshold,
        )
        return {"x": x_train, "y": y_train}, {"x": x_test, "y": y_test}


class DGPOpenML(DGPReal):
    """
    Download data from OpenML and identify the target and feature types.
    """

    openml_id: int
    target_index: int
    target_name: str
    feature_name: list[str]

    def __init__(
        self, openml_id: int, target_index: int, categorical_x: list[bool] | None
    ):
        self.openml_id = openml_id
        self.target_index = target_index
        dataset = openml.datasets.get_dataset(openml_id)

        # Get the data itself as a dataframe (or otherwise)
        df, _, is_categorical, var_names = dataset.get_data(dataset_format="dataframe")

        if categorical_x is None:
            # If not provided, use the OpenML's categorical information
            del is_categorical[target_index]  # remove the target column
            categorical_x = is_categorical

        self.target_name = var_names[target_index]
        logging.info(f"OpenML ID: {openml_id}, target: {self.target_name}")

        # Multiclass yeast and wine are hugely imbalance. Filter out the classes
        # with little observations (<3% of the overall).
        if openml_id in [40498, 181]:
            class_proportion = df[self.target_name].value_counts(normalize=True)
            # total_observations = len(X)
            classes_to_remove = class_proportion[class_proportion < 0.03].index
            df = df[~df[self.target_name].isin(classes_to_remove)]

        # Separate the target column and the rest of the columns
        y_column = df[[self.target_name]].to_numpy().squeeze()
        x_columns = df.drop(columns=[self.target_name]).to_numpy()

        self.categorical_x = categorical_x
        var_names.pop(target_index)
        self.feature_name = var_names
        self.full_data = {"x": x_columns, "y": y_column}


class DGPUCI(DGPReal):
    """
    Download data from UCI and identify the target and feature types.
    """

    uci_id: int
    target_name: str
    feature_name: list[str]

    def __init__(self, uci_id: int, categorical_x: list[bool]):
        self.uci_id = uci_id
        if not os.path.exists(f"datasets/uci-{uci_id}.pickle"):
            dataset = fetch_ucirepo(id=uci_id)
            utils.write_to(f"datasets/uci-{uci_id}.pickle", dataset)
        else:
            dataset = utils.read_from(f"datasets/uci-{uci_id}.pickle")
        variables = dataset.variables
        self.target_name = variables.loc[variables["role"] == "Target", "name"].iloc[0]
        self.categorical_x = categorical_x

        logging.info(f"UCI ID: {uci_id}, target: {self.target_name}")
        X = dataset.data.features.to_numpy()
        y = dataset.data.targets.to_numpy().squeeze()
        assert X.shape[1] == len(categorical_x)
        self.feature_name = variables.loc[
            variables["role"] == "Feature", "name"
        ].to_list()
        self.full_data = {"x": X, "y": y}


class DGPRegressionOpenML(DGPOpenML):

    def __init__(
        self,
        key: PRNGKeyArray,
        n: int,
        openml_id: int,
        target_index: int,
        strata_bins: int = 1,
        categorical_x: list[bool] | None = None,
        continuous_threshold: int | None = None,
    ):
        super().__init__(openml_id, target_index, categorical_x)
        self.input_key = key
        self.strata_bins = strata_bins
        logging.debug(f"Strata bins: {strata_bins}")
        self.train_data, self.test_data = self.split_data(
            key, n, is_y_categorical=False, continuous_threshold=continuous_threshold
        )


class DGPClassificationOpenML(DGPOpenML):

    def __init__(
        self,
        key: PRNGKeyArray,
        n: int,
        openml_id: int,
        target_index: int,
        strata_bins: int = 1,
        categorical_x: list[bool] | None = None,
        continuous_threshold: int | None = None,
    ):
        super().__init__(openml_id, target_index, categorical_x)
        self.input_key = key
        self.strata_bins = strata_bins
        logging.debug(f"Strata bins: {strata_bins}")
        self.train_data, self.test_data = self.split_data(
            key, n, is_y_categorical=True, continuous_threshold=continuous_threshold
        )


class DGPClassificationUCI(DGPUCI):

    def __init__(
        self,
        key: PRNGKeyArray,
        n: int,
        uci_id: int,
        categorical_x: list[bool],
        strata_bins: int = 1,
    ):
        super().__init__(uci_id, categorical_x)
        self.input_key = key
        self.strata_bins = strata_bins
        logging.debug(f"Strata bins: {strata_bins}")
        self.train_data, self.test_data = self.split_data(key, n, is_y_categorical=True)


class DGPLidar(DGPReal):

    def __init__(self, key: PRNGKeyArray, n: int):
        self.input_key = key

        DATA_URI = "http://www.stat.cmu.edu/~larry/all-of-nonpar/=data/lidar.dat"

        df = pd.read_csv(DATA_URI, sep=r"\s+")
        y = df["logratio"].values
        x = df["range"].values.reshape(-1, 1)

        logging.info(f"LIDAR, target: logratio")
        self.full_data = {"x": x, "y": y}
        self.categorical_x = [False]  # LIDAR has no categorical features
        self.strata_bins = 1

        # Convert to NumPy arrays
        if n == -1:
            # If n is -1, use the full dataset as training data
            self.train_data = {"x": x, "y": y}
            self.test_data = {"x": x}
        else:
            self.train_data, self.test_data = self.split_data(
                key, n, is_y_categorical=False
            )


class DGPGeneratorBacked(DGP):
    generator: object
    dim_x: int
    test_data: dict[str, np.ndarray] | None
    categorical_x: list[bool]
    target_name: str
    feature_name: list[str]
    metadata: dict[str, object]

    def __init__(self, key: PRNGKeyArray, n: int, generator: object, test_data_size: int = 0):
        self.input_key = key
        self.generator = generator
        self.dim_x = generator.num_features
        self.categorical_x = list(generator.categorical_x)
        self.target_name = generator.target_name
        self.feature_name = list(generator.feature_name)
        self.metadata = dict(getattr(generator, "metadata", {}))
        key, data_key, test_key = jax.random.split(key, 3)
        self.train_data = self.get_data(data_key, n)
        self.test_data = (
            self.get_data(test_key, test_data_size) if test_data_size > 0 else None
        )

    def get_x_data(self, key: PRNGKeyArray, n: int) -> Array:
        return self.generator.sample_x(key, n)

    def get_data(self, key: PRNGKeyArray, n: int) -> dict[str, np.ndarray]:
        sample = self.generator.sample(key, n)
        if "metadata" in sample and not self.metadata:
            self.metadata = dict(sample["metadata"])
        return {
            "x": np.asarray(sample["x"], dtype=np.float64),
            "y": np.asarray(sample["y"]),
        }


class DGPClassificationFixed(DGPGeneratorBacked):
    def __init__(
        self, key: PRNGKeyArray, n: int, dim_x: int, test_data_size: int = 0
    ):
        super().__init__(
            key,
            n,
            ClassificationFixedGenerator(dim_x),
            test_data_size=test_data_size,
        )


class DGPClassificationFixedGMMLink(DGPGeneratorBacked):
    def __init__(
        self,
        key: PRNGKeyArray,
        n: int,
        dim_x: int,
        a: float,
        test_data_size: int = 0,
    ):
        super().__init__(
            key,
            n,
            ClassificationFixedGMMLinkGenerator(dim_x, a),
            test_data_size=test_data_size,
        )


class DGPClassificationLinear(DGPGeneratorBacked):
    num_classes: int

    def __init__(
        self,
        key: PRNGKeyArray,
        n: int,
        dim_x: int,
        num_classes: int,
        test_data_size: int = 0,
    ):
        param_key = jax.random.split(key, 2)[0]
        param_seed = int(
            jax.random.randint(
                param_key, shape=(), minval=0, maxval=2_147_483_647
            ).item()
        )
        self.num_classes = num_classes
        super().__init__(
            key,
            n,
            ClassificationLinearGenerator(dim_x, num_classes, seed=param_seed),
            test_data_size=test_data_size,
        )
        classes, counts = np.unique(self.train_data["y"], return_counts=True)
        logging.info(
            "Linear classification generated: "
            f"n_features={dim_x}, "
            f"requested_classes={num_classes}, "
            f"observed_classes={classes.tolist()}, "
            f"class_counts={counts.tolist()}"
        )


class DGPRegressionFixed(DGPGeneratorBacked):
    def __init__(
        self,
        key: PRNGKeyArray,
        n: int,
        dim_x: int,
        noise_std: float,
        test_data_size: int = 0,
    ):
        super().__init__(
            key,
            n,
            RegressionFixedGenerator(dim_x, noise_std),
            test_data_size=test_data_size,
        )


class DGPRegressionFixedDependentError(DGPGeneratorBacked):
    def __init__(
        self,
        key: PRNGKeyArray,
        n: int,
        dim_x: int,
        s_small: float,
        s_mod: float,
        test_data_size: int = 0,
    ):
        super().__init__(
            key,
            n,
            RegressionFixedDependentErrorGenerator(dim_x, s_small, s_mod),
            test_data_size=test_data_size,
        )


class DGPRegressionFixedNonNormalError(DGPGeneratorBacked):
    def __init__(
        self, key: PRNGKeyArray, n: int, dim_x: int, df: int, test_data_size: int = 0
    ):
        super().__init__(
            key,
            n,
            RegressionFixedNonNormalErrorGenerator(dim_x, df),
            test_data_size=test_data_size,
        )


class DGPClassificationSCMNN(DGPGeneratorBacked):
    num_classes: int

    def __init__(
        self,
        key: PRNGKeyArray,
        n: int,
        dim_x: int,
        num_classes: int,
        test_data_size: int = 0,
    ):
        param_key = jax.random.split(key, 2)[0]
        param_seed = int(
            jax.random.randint(
                param_key, shape=(), minval=0, maxval=2_147_483_647
            ).item()
        )
        self.num_classes = num_classes
        super().__init__(
            key,
            n,
            SCMNNClassificationGenerator(dim_x, num_classes, seed=param_seed),
            test_data_size=test_data_size,
        )
        classes, counts = np.unique(self.train_data["y"], return_counts=True)
        logging.info(
            "SCM-NN classification generated: "
            f"n_features={dim_x}, "
            f"requested_classes={num_classes}, "
            f"observed_classes={classes.tolist()}, "
            f"class_counts={counts.tolist()}"
        )


class DGPClassificationSCMDag(DGPGeneratorBacked):
    num_classes: int
    max_parents: int

    def __init__(
        self,
        key: PRNGKeyArray,
        n: int,
        dim_x: int,
        num_classes: int,
        max_parents: int = 2,
        test_data_size: int = 0,
    ):
        param_key = jax.random.split(key, 2)[0]
        param_seed = int(
            jax.random.randint(
                param_key, shape=(), minval=0, maxval=2_147_483_647
            ).item()
        )
        self.num_classes = num_classes
        self.max_parents = max_parents
        super().__init__(
            key,
            n,
            SCMDagClassificationGenerator(
                dim_x, num_classes, seed=param_seed, max_parents=max_parents
            ),
            test_data_size=test_data_size,
        )
        classes, counts = np.unique(self.train_data["y"], return_counts=True)
        logging.info(
            "SCM-DAG classification generated: "
            f"n_features={dim_x}, "
            f"requested_classes={num_classes}, "
            f"max_parents={max_parents}, "
            f"observed_classes={classes.tolist()}, "
            f"class_counts={counts.tolist()}"
        )


class DGPRegression(DGPGeneratorBacked):
    def __init__(
        self, key: PRNGKeyArray, n: int, dim_x: int, test_data_size: int = 0
    ):
        key, prior_key = jax.random.split(key)
        super().__init__(
            key,
            n,
            RegressionPriorGenerator(prior_key, dim_x),
            test_data_size=test_data_size,
        )


class DGPClassification(DGPGeneratorBacked):
    def __init__(
        self, key: PRNGKeyArray, n: int, dim_x: int, test_data_size: int = 0
    ):
        key, prior_key = jax.random.split(key)
        super().__init__(
            key,
            n,
            ClassificationPriorGenerator(prior_key, dim_x),
            test_data_size=test_data_size,
        )


class DGPLinearRegressionWM(DGPGeneratorBacked):
    def __init__(self, key: PRNGKeyArray, n: int, test_data_size: int = 0):
        super().__init__(
            key,
            n,
            LinearRegressionWMGenerator(),
            test_data_size=test_data_size,
        )


class DGPDependentErrorWM(DGPGeneratorBacked):
    def __init__(
        self, key: PRNGKeyArray, n: int, s_small: float, s_mod: float, test_data_size: int = 0
    ):
        super().__init__(
            key,
            n,
            DependentErrorWMGenerator(s_small, s_mod),
            test_data_size=test_data_size,
        )


class DGPNonNormalErrorWM(DGPGeneratorBacked):
    def __init__(self, key: PRNGKeyArray, n: int, df: int, test_data_size: int = 0):
        super().__init__(
            key,
            n,
            NonNormalErrorWMGenerator(df),
            test_data_size=test_data_size,
        )


OPENML_REGRESSION = [
    "abalone",
    "airfoil",
    "kin8nm",
    "auction",
    "concrete",
    "energy",
    "grid",
    "fish",
    "quake",
]

OPENML_BINARY_CLASSIFICATION = [
    "blood",
    "phoneme",
    "banknote",
    "mozilla",
    "skin",
    "telescope",
    "sepsis",
    "rice",
]

OPENML_CLASSIFICATION = [
    "vehicle",
    "yeast",
    "wine",
]


def load_dgp(cfg, data_key: PRNGKeyArray) -> DGP:
    synthetic_test_data_size = getattr(cfg, "synthetic_test_data_size", 0)

    if cfg.dgp.name == "classification-fixed":
        dgp = DGPClassificationFixed(
            data_key, cfg.data_size, cfg.dgp.dim_x, synthetic_test_data_size
        )
    elif cfg.dgp.name == "classification-fixed-gmm":
        dgp = DGPClassificationFixedGMMLink(
            data_key,
            cfg.data_size,
            cfg.dgp.dim_x,
            cfg.dgp.a,
            synthetic_test_data_size,
        )
    elif cfg.dgp.name == "classification-linear":
        dgp = DGPClassificationLinear(
            data_key,
            cfg.data_size,
            cfg.dgp.dim_x,
            cfg.dgp.num_classes,
            synthetic_test_data_size,
        )
    elif cfg.dgp.name == "classification-scm_nn":
        dgp = DGPClassificationSCMNN(
            data_key,
            cfg.data_size,
            dim_x=cfg.dgp.dim_x,
            num_classes=cfg.dgp.num_classes,
            test_data_size=synthetic_test_data_size,
        )
    elif cfg.dgp.name == "classification-scm_dag":
        dgp = DGPClassificationSCMDag(
            data_key,
            cfg.data_size,
            dim_x=cfg.dgp.dim_x,
            num_classes=cfg.dgp.num_classes,
            max_parents=getattr(cfg.dgp, "max_parents", 2),
            test_data_size=synthetic_test_data_size,
        )
    elif cfg.dgp.name == "regression-fixed":
        dgp = DGPRegressionFixed(
            data_key,
            cfg.data_size,
            cfg.dgp.dim_x,
            cfg.dgp.noise_std,
            synthetic_test_data_size,
        )
    elif cfg.dgp.name == "regression-fixed-dependent":
        dgp = DGPRegressionFixedDependentError(
            data_key,
            cfg.data_size,
            cfg.dgp.dim_x,
            s_small=cfg.dgp.s_small,
            s_mod=cfg.dgp.s_mod,
            test_data_size=synthetic_test_data_size,
        )
    elif cfg.dgp.name == "regression-fixed-non-normal":
        dgp = DGPRegressionFixedNonNormalError(
            data_key,
            cfg.data_size,
            cfg.dgp.dim_x,
            df=cfg.dgp.df,
            test_data_size=synthetic_test_data_size,
        )
    elif cfg.dgp.name == "regression-wm":
        dgp = DGPLinearRegressionWM(data_key, cfg.data_size, synthetic_test_data_size)
    elif cfg.dgp.name == "dependent-error-wm":
        dgp = DGPDependentErrorWM(
            data_key,
            cfg.data_size,
            s_small=cfg.dgp.s_small,
            s_mod=cfg.dgp.s_mod,
            test_data_size=synthetic_test_data_size,
        )
    elif cfg.dgp.name == "non-normal-wm":
        dgp = DGPNonNormalErrorWM(
            data_key,
            cfg.data_size,
            df=cfg.dgp.df,
            test_data_size=synthetic_test_data_size,
        )
    elif cfg.dgp.name == "quake":
        dgp = DGPRegressionOpenML(data_key, cfg.data_size, 550, -1, 2)
    elif cfg.dgp.name == "airfoil":
        dgp = DGPRegressionOpenML(data_key, cfg.data_size, 44957, -1, 1)
    elif cfg.dgp.name == "kin8nm":
        dgp = DGPRegressionOpenML(data_key, cfg.data_size, 44980, -1, 1)
    elif cfg.dgp.name == "concrete":
        dgp = DGPRegressionOpenML(data_key, cfg.data_size, 44959, -1, 1)
    elif cfg.dgp.name == "energy":
        if cfg.data_size > 50:
            dgp = DGPRegressionOpenML(data_key, cfg.data_size, 44960, -1, 1)
        else:
            dgp = DGPRegressionOpenML(data_key, cfg.data_size, 44960, -1, 1, None, 2)
    elif cfg.dgp.name == "grid":
        dgp = DGPRegressionOpenML(data_key, cfg.data_size, 44973, -1, 1)
    elif cfg.dgp.name == "abalone":
        dgp = DGPRegressionOpenML(data_key, cfg.data_size, 45042, -1, 1)
    elif cfg.dgp.name == "fish":
        if cfg.data_size > 50:
            dgp = DGPRegressionOpenML(data_key, cfg.data_size, 44970, -1, 2)
        else:
            dgp = DGPRegressionOpenML(data_key, cfg.data_size, 44970, -1, 1)
    elif cfg.dgp.name == "auction":
        if cfg.data_size > 50:
            dgp = DGPRegressionOpenML(data_key, cfg.data_size, 44958, -1, 1)
        else:
            dgp = DGPRegressionOpenML(data_key, cfg.data_size, 44958, -1, 1, None, 1)
    elif cfg.dgp.name == "blood":
        dgp = DGPClassificationOpenML(data_key, cfg.data_size, 1464, -1, 1)
    elif cfg.dgp.name == "phoneme":
        dgp = DGPClassificationOpenML(data_key, cfg.data_size, 1489, -1, 1)
    elif cfg.dgp.name == "skin":
        dgp = DGPClassificationOpenML(data_key, cfg.data_size, 1502, -1, 3)
    elif cfg.dgp.name == "rice":
        if cfg.data_size > 50:
            dgp = DGPClassificationUCI(data_key, cfg.data_size, 545, [False] * 7, 2)
        else:
            dgp = DGPClassificationUCI(data_key, cfg.data_size, 545, [False] * 7, 1)
    elif cfg.dgp.name == "mozilla":
        dgp = DGPClassificationOpenML(data_key, cfg.data_size, 1046, -1, 2)
    elif cfg.dgp.name == "telescope":
        dgp = DGPClassificationUCI(data_key, cfg.data_size, 159, [False] * 10, 1)
    elif cfg.dgp.name == "sepsis":
        dgp = DGPClassificationUCI(
            data_key, cfg.data_size, 827, [False, True, False], 2
        )
    elif cfg.dgp.name == "yeast":
        dgp = DGPClassificationOpenML(data_key, cfg.data_size, 181, -1, 1)
    elif cfg.dgp.name == "vehicle":
        dgp = DGPClassificationOpenML(data_key, cfg.data_size, 54, -1, 1)
    elif cfg.dgp.name == "wine":
        dgp = DGPClassificationOpenML(data_key, cfg.data_size, 40498, -1, 1)
    elif cfg.dgp.name == "banknote":
        if cfg.data_size > 50:
            dgp = DGPClassificationOpenML(data_key, cfg.data_size, 1462, -1, 2)
        else:
            dgp = DGPClassificationOpenML(data_key, cfg.data_size, 1462, -1, 1)
    else:
        raise NotImplementedError(f"DGP {cfg.dgp.name} not implemented")
    return dgp
