"""
############################
Subsequence Identification
############################
This package provides the Autoencoder based iterative modeling and multivariate time-series subsequence clustering algorithm (ABIMCA).

Features
###########################

- Unsupervised clustering of multivariate time-series data.

"""

# Standard Libraries
import os
import random
import time
from collections import deque
from pathlib import Path
from typing import Callable

# Third Party Libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import signal, stats
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.decomposition import PCA
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Local Scripts
from .autoencoders import (
    AutoEncoder,
    BidirectionalGruAutoEncoder,
    GRUAutoEncoder,
    set_optimizer,
    setWeightInitialization,
)
from .losses import RMSELoss, SparseMSELoss, SparseRMSELoss

LOGGER_NAME = "abimca"


class SubsequenceIdentifier(ClusterMixin, BaseEstimator):
    """[summary]

    .. important::

        Since the subsequence identification mehtod is using rolling time windows, it is crucial to not have any time windows which overlap between instances. Therefore the 'partial_fit' function needs to be used, when more than one instance is used for training data. One instance is defined here as one measurement, with consistent consecutive timestamps. (No gap between datapoints in time should be existing here!)

        Attention: theta_factor has a higher priority than theta. If theta_factor is not None this is used instead of theta!

    Args:
        ClusterMixin ([type]): [description]
        BaseEstimator ([type]): [description]
    """

    def __init__(
        self,
        seq_len: int = 10,
        eta: float = 0.05,
        phi: int = 10,
        theta: float = 0.1,
        omega: int = 10,
        psi: int = 5,
        dim_bottleneck: int = None,
        learning_rate: float = 1e-2,
        step_size: int = 1,
        ae_type: str = "BidirectionalGruAutoEncoder",
        optimizer_type: str = "SGD",
        disable_progress_bar: bool = True,
        save_training_results=False,
        reuse_optimizer=True,
        allow_unknown=True,
        use_lowest_loss=True,
        set_parameters_automatically=True,
        traindata_contains_subsequences_multiple_times=False,  # BUG: Rename this!
        use_denoising_ae=False,
        noise_std=0.2,
        scoring_method: str = "mix",  # "reconstruction_loss", "latent_mean", "mix"
        mix_weighting_factor: float = 1,
        tempfolder_suffix="",
        theta_factor: float = None,  # Attention: theta_factor has a higher priority than theta. If theta_factor is not None this is used instead of theta!
        momentum: float = 0.9,
        use_weight_init: str = "sparse",  #  "constant", "kaiming_uniform"
        seed: int = 42,
        **kwargs,
    ) -> None:
        self.seq_len = seq_len
        self.eta = eta  # new sequence loss threshold
        self.phi = phi  # nr of score mean values for base autoencoder (should be larger than psi)
        self.theta_factor = theta_factor  # Attention: theta_factor has a higher priority than theta. If theta_factor is not None this is used instead of theta!
        if theta_factor is not None:
            self.theta = eta * theta_factor
            # print(f"setting theta: with {eta=} and {theta_factor=} to {self.theta=}")
        else:
            self.theta = theta  # min models loss threshold for subsequence recognition
        self.omega = omega  # base model overfitrate
        self.psi = psi  # nr of samples per moving average loss for subsequence models (this should be smaller than phi; Why different to psi? Because then subsequences can be better recognized before a new subsequence is created due to timing issues. Otherwise the base ae is might be quicker under the threshold than the subsequence model)
        self.dim_bottleneck = dim_bottleneck  # The bottleneck dimension of the autoencoder. This should not bee too small. The intrinsic dimension is a good starting point. Intrinsic dimension could be obtained from a PCA for example.
        self.learning_rate = learning_rate  # The learning rate of the optimizer
        self.step_size = step_size  # If not every step should be used during training. THis can speed up training but reduces data used. For example if set to 2, only every second datapoint is used for training.
        self.ae_type: str = ae_type  # Choose the type of autoencoder used for the algorithms subsequence models.
        self.optimizer_type = optimizer_type  # Choose the tpe of optimizer to use. Currently implemented only `Adam` or `AdamW`
        self.disable_progress_bar = disable_progress_bar  # Disables the progress bar during training. Useful for Hyperparameter optimization.
        self.save_training_results = save_training_results  # Saves the training results for later use like plotting. (This can help set the parameters better)
        self.reuse_optimizer = reuse_optimizer  # This flag resets the optimizer during every training step. The optimizer is then only keeping parameters for the amount of iterations set with 'omega' parameter.
        self.allow_unknown = allow_unknown  # If this is set, during prediction known subsequence models need to be lower than threshold (theta) to be identified as a subsequence. All datapoints over the threshold are classified as class 0.
        self.use_lowest_loss = use_lowest_loss  # If set this will save the model for the current subsequence again if the loss is lower than a previous point. So if during a recognized subsequence the loss is still going down (e.g: due to a low learning rate) the model for this subsequence is not used from the point of falling below the threshold (eta) but from the point with the lowest loss of this subsequence.
        self.set_parameters_automatically = set_parameters_automatically  # Trying to set eta and theta automatically based on minimal loss for randomized training data training.
        self.traindata_contains_subsequences_multiple_times = traindata_contains_subsequences_multiple_times  # #TODO: rename this! Dropping new subsequence model during training if previous model has low loss. Only set this to 'True' if you are sure, that each subsequence exists only once in the training data.
        self.use_denoising_ae = use_denoising_ae
        self.noise_std = noise_std
        self.scoring_method = scoring_method  # Define the scoring method that shall be used for comparing to threshold and identify the time step as a subsequence or not. This needs to be a string of one of the following: "reconstruction_loss" (uses the MSE of the Autoencoders reconstruction loss), "latent_mean" (This uses the mean value of the latent variables), "mix" (Uses reconstruction loss and mean of latent variables)
        self.mix_weighting_factor = mix_weighting_factor  # If scoring_method == "mix" then this multiplies latent_mean value and divides reconstruction loss. So if set to 2 then latent mean is 4 times more influential than the reconstruction error.
        self.tempfolder_suffix = tempfolder_suffix
        self.momentum = momentum  # The moment if SGD optimizer is chosen
        self.use_weight_init = use_weight_init
        self.seed = seed
        # print(f"ABIMCA Init with params: {use_weight_init=} {momentum=} {optimizer_type=}.")

    def setup_fitting(self):
        logger.debug(f"Setup fitting..")
        # TODO: Attach '_' to end of all of these parameters!!
        self._set_seed()
        self.dataloader_shuffle = False
        self.num_workers = 1
        self.batch_size = 1
        self.predict_batch_size = 2000
        self.pin_memory = False
        self.loss_function_name: str = "SparseMSELoss"
        self.root = Path(os.path.dirname(__file__))
        self.temp_folder = self.root / str(".abimca_tmp" + self.tempfolder_suffix)
        self.pretrained_base_model_file_path = self.temp_folder / "base_model.pt"
        self.buffered_results_path = self.temp_folder / "buff_results.pkl"
        self.buffered_prediction_results_path = (
            self.temp_folder / "buff_prediction_results.pkl"
        )
        self.buffered_reconstruction_path = self.temp_folder / "buff_reconstruction.pkl"
        self.current_seq_counter = 0
        self.adapt_learning_rate = False
        self.load_pretrained_base_model_from_file = False
        self.pretrain_base_model = True
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.list_models = []
        self.lowest_subs_id_losses = []
        self._saved_training_results = False
        self.dim_bottleneck = self._set_dim_bottleneck()
        self.base_model = self._set_model(self.ae_type)
        self.loss_fn = self._set_loss_function(self.loss_function_name)
        self.optimizer = self._set_optimizer()
        self.reconstruction_list = []
        self.alldata_list = []
        self.alldata_columns = [
            "model",
            "loss",
            "step",
            "subs_id",
            "seq_models_counter",
            "latent_mean",
        ]
        self.adapt_factor_ = 2
        self.reuse_base_model_from_automatically_parameter_setting_ = True
        self.is_fitted_ = False
        self.locations = None

    def fit(self, X: np.array, y: np.array = None):
        # y are the instance labels if instances exist in the data provided
        if y is None:
            # If no instance labels, then we assume all data points are from one instances and have consecutive time steps.
            # Generateing constant array with zeros of shape of X
            y = np.full(X.shape[0], 0)
        # Validate the data
        X, y = check_X_y(X, y, accept_sparse=False)
        self.num_features = X.shape[1]
        self._num_original_time_steps = X.shape[0]
        # Setup fitting
        self.setup_fitting()
        # Create dataset with rolling windows of 'seq_len' length
        dataset = MultipleTimeStepsDataset(
            X, y, seq_len=self.seq_len, step_size=self.step_size
        )

        if self.set_parameters_automatically is True:
            logger.info("Setting the parameters automatically..")
            new_params = self._pre_data_analysis(X, dataset)
            logger.debug(f"Resetting the following parameters {new_params}")
            self.set_params(**new_params)
            self.base_model = self._set_model(self.ae_type)
            self.optimizer = self._set_optimizer()
        if self.pretrain_base_model is True:
            logger.info("Pretraining the base model..")
            lrf = LearningRateFinder(
                X,
                self.base_model,
                self.loss_fn,
                self.optimizer_type,
                self.device,
                self.seq_len,
                score_fn=self._score_fn,
                default_lr=self.learning_rate,
                seed=self.seed,
            )
            self.base_model = lrf.pretrain(
                disable_progress_bar=self.disable_progress_bar
            )
        eta = self.eta  # new sequence loss threshold
        theta = self.theta  # min models loss threshold for subsequence recognition
        dataloader = DataLoader(
            dataset,
            self.batch_size,
            shuffle=self.dataloader_shuffle,
            pin_memory=self.pin_memory,
            collate_fn=None,
            num_workers=self.num_workers,
            worker_init_fn=_init_fn,
            generator=torch_gen,
        )
        # Instantiate all necessary variables
        if self.reuse_base_model_from_automatically_parameter_setting_ is False:
            self.base_model = self._set_model(self.ae_type)
        self.movingaverages = (
            []
        )  # List for collecting the moving averages of each subsequence models loss
        phi = self.phi  # nr of indicator mean values
        subs_id = 0  # subsequence id
        S_ids = []
        sc_ma = MovingAverage(size=phi)
        self.base_model
        # Start train loop
        with tqdm(total=len(dataset), disable=self.disable_progress_bar) as bar:
            for step, data in enumerate(dataloader):
                scores_np = None
                x = data[0].to(self.device)
                start = time.time()
                # Algorithm action here:
                # Give new data point to base model and perform training step
                x_orig, x_hat, latent, avg_loss = self.base_model_train_step(x)
                score = self._score_fn(avg_loss, latent.mean())
                score_ma = sc_ma.update(score)
                # subs_id, scores_np = self.set_sequence_id(x, theta, step)
                scores_np = self._get_score_of_models_for_known_sequences(
                    self.list_models, x
                )
                if (
                    scores_np is not None and scores_np.min() < theta
                ):  # Subsequence recognized
                    if self.traindata_contains_subsequences_multiple_times is True:
                        if (
                            (len(scores_np) > 1)
                            and (scores_np[:-1].min() < self.eta)
                            and (scores_np[-1] < self.eta)
                        ):
                            self._drop_last_model()
                            scores_np = self._get_score_of_models_for_known_sequences(
                                self.list_models, x
                            )
                    # Set ID to recognized subsequence model (model with lowest score)
                    subs_id = scores_np.argmin() + 1
                    if self.use_lowest_loss is True:
                        self._compare_current_subs_loss_with_base_model_loss(
                            subs_id, scores_np, score_ma
                        )
                else:  # No subsequence recognized setting ID to -1
                    if score_ma <= eta:  # Score below new subsequence threshold
                        subs_id, scores_np = self._set_new_sequence_model(
                            scores_np, score_ma
                        )  # Setting new model
                    else:  # We are in a transition phase but threshold for setting new subsequence model (eta) is not reached
                        subs_id = 0  # Therefore setting ID to zero (unknown sequence)
                bar.update()
                ende = time.time()
                t_trainloop = ende - start
                S_ids.append(subs_id)

                if self.save_training_results is True:
                    start_save = time.time()
                    self._saved_training_results = self._buffer_training_results(
                        x_orig, x_hat, latent, score_ma, step, scores_np, subs_id
                    )
                    ende_save = time.time()
                    t_saving = ende_save - start_save
                    bar.set_description(
                        f"Step: {step}, Nr of models: {len(self.list_models)} subs_id {subs_id} t_trainloop: {t_trainloop:.2f}s seq_counter {self.current_seq_counter} base loss {score_ma:.4f} time spend for buffering intermediate results: {t_saving:.4f}s"
                    )
                else:
                    bar.set_description(
                        f"Step: {step}, Nr of models: {len(self.list_models)} subs_id {subs_id} t_trainloop: {t_trainloop:.2f} seq_counter {self.current_seq_counter} base loss {score_ma:.4f}"
                    )
                # TODO: here i could implement a rau tune.report(metric) with number of subsequences for early trial stopping as well as timing??
                # bar.set_description(f"Step: {step}, t_trainloop: {t_trainloop:.2f}")
        self._saved_training_results = self._save_buffered_results()
        self.label_array = np.array(S_ids)
        if self.step_size > 1:
            self.label_array = np.repeat(self.label_array, self.step_size, axis=0)
            self.label_array = self.label_array[: X.shape[0]]
        locations = list(np.where(self.label_array[:-1] != self.label_array[1:])[0])
        if self.current_seq_counter == 0:
            self.is_fitted_ = False
            self.found_subsequences_ = False
            self.locations = []
        else:
            self.is_fitted_ = True
            self.found_subsequences_ = True
            self.locations = locations
        return self

    def predict(self, X: np.array, y: np.array = None):
        """A reference implementation of a prediction for a classifier.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        y : ndarray, shape (n_samples,)
            The label for each sample is the label of the closest sample
            seen during fit.
        """
        # Check is fit had been called
        start_predict = time.time()
        check_is_fitted(self, "found_subsequences_")

        # Input validation
        if y is None:
            # If no instance labels, then we assume all data points are from one instances and have consecutive time steps.
            # Generateing constant array with zeros of shape of X
            y = np.full(X.shape[0], 0)
        X, y = check_X_y(X, y, accept_sparse=False)

        # Create dataset with rolling windows of 'seq_len' length
        dataset = MultipleTimeStepsDataset(
            X, y, seq_len=self.seq_len, device=self.device, step_size=self.step_size
        )
        dataloader = DataLoader(
            dataset,
            self.predict_batch_size,
            shuffle=self.dataloader_shuffle,
            pin_memory=self.pin_memory,
            collate_fn=None,
            num_workers=self.num_workers,
            worker_init_fn=_init_fn,
            generator=torch_gen,
        )
        # Start inference loop
        results = []
        with tqdm(total=len(dataloader), disable=self.disable_progress_bar) as bar:
            with torch.no_grad():
                for step, data in enumerate(dataloader):
                    x = data[0].to(self.device)
                    # scores = self._get_score_of_models_for_known_sequences_as_batch(self.list_models, x)
                    scores = self._get_score_of_models_for_known_sequences(
                        self.list_models, x
                    )
                    results.append(scores)
                    bar.update()
        try:
            results = np.concatenate(results, axis=1).T
            # Doing the moving average here over all data with a rolling window! No instances are considered here!
            results = (
                pd.DataFrame(results).rolling(self.psi, min_periods=1).mean().values
            )
        except Exception:
            logger.warning(f"Empty class prediction. Setting it all to zero!")
            results = np.full((X.shape[0], 1), 0)
        if self.step_size > 1:
            # import pdb;pdb.set_trace()
            results = np.repeat(results, self.step_size, axis=0)
            results = results[: X.shape[0]]
        seq_idx = np.apply_along_axis(self.calc_seq_idx, axis=1, arr=results)
        self._prediction_losses = results
        self._saved_prediction_results = self._save_buffered_prediction_results()
        ende = time.time()
        t_predict = ende - start_predict
        logger.debug(f"Prediction time in seconds: {t_predict}")
        return seq_idx

    def calc_seq_idx(self, x):
        if self.allow_unknown is True:
            min_val = x.min()
            if min_val < self.theta:
                idx = x.argmin() + 1
            else:
                idx = 0
        else:
            idx = x.argmin() + 1
        return idx

    def base_model_train_step(self, x: torch.tensor):
        alllosses = []
        if self.reuse_optimizer is True:
            self.optimizer = self._set_optimizer()
        for _ in np.arange(self.omega):
            # Add noise:
            if self.use_denoising_ae is True:
                mean = 0
                std = self.noise_std
                noise = np.random.normal(mean, std, x.shape)
                noise = torch.tensor(noise, dtype=torch.float32).to(self.device)
                x_train = x + noise
            else:
                x_train = x
            y_pred, latent = self.base_model(x_train)
            loss = self.loss_fn(y_pred, x, latent)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            alllosses.append(loss.cpu().detach().numpy())
        avg_loss = np.array(alllosses).mean()
        x_orig = np.expand_dims(x[0, -1, :].cpu().detach().numpy(), axis=0)
        x_hat = y_pred[:, -1, :].cpu().detach().numpy()
        latent = latent[:, -1, :].cpu().detach().numpy()
        return x_orig, x_hat, latent, avg_loss

    def set_sequence_id(
        self, x: torch.tensor, theta: float, step: int
    ) -> tuple[int, np.array]:
        scores_np = self._get_score_of_models_for_known_sequences(self.list_models, x)
        if scores_np.min() < theta:  # Subsequence recognized
            if self.traindata_contains_subsequences_multiple_times is True:
                if (
                    (len(scores_np) > 1)
                    and (scores_np[:-1].min() < self.eta)
                    and (scores_np[-1] < self.eta)
                ):
                    self._drop_last_model()
                    scores_np = self._get_score_of_models_for_known_sequences(
                        self.list_models, x
                    )
            # Set ID to recognized subsequence model (model with lowest score)
            subs_id = scores_np.argmin() + 1
        else:
            # No subsequence recognized setting ID to -1
            subs_id = -1
        return (subs_id, scores_np)

    def set_sequence_id_old(
        self, x: torch.tensor, theta: float, step: int
    ) -> tuple[int, np.array]:
        subs_id = -1
        scores_np = None
        scores_min = None
        found_subsequence_again = None
        if len(self.list_models) > 0:
            scores_np = self._get_score_of_models_for_known_sequences(
                self.list_models, x
            )
            scores_min = scores_np.min()
            if scores_np.min() < theta:
                subs_id = scores_np.argmin() + 1
                if self.traindata_contains_subsequences_multiple_times is True:
                    if (
                        (len(scores_np) > 1)
                        and (scores_np[:-1].min() < self.eta)
                        and (scores_np[-1] < self.eta)
                    ):
                        self._drop_last_model()
                        subs_id = scores_np[:-1].argmin() + 1
                        found_subsequence_again = True
                    else:
                        found_subsequence_again = False
        if subs_id == -1:
            pass
            # logger.debug(f"Setting new model because: {scores_min=} > {theta=} and {found_subsequence_again=}")

        return (subs_id, scores_np)

    def _set_new_sequence_model(self, scores_np: np.array, loss_ma: float):
        self.movingaverages.append(MovingAverage(size=self.psi))
        subseq_model = self._set_model(self.ae_type)
        subseq_model.load_state_dict(self.base_model.state_dict())
        self.list_models.append(subseq_model)
        self.lowest_subs_id_losses.append(loss_ma)
        S_id = len(self.list_models)
        self.current_seq_counter += 1
        if scores_np is not None:
            scores_np = np.append(scores_np, loss_ma)
        else:
            scores_np = np.array([loss_ma])
        # self.save_subsequence_model(subseq_model, S_id)
        return S_id, scores_np

    def _score_fn(
        self, avg_loss: float, latent: float, eps: float = 1e-10
    ) -> np.float64:
        if self.scoring_method == "latent_mean":
            score = np.absolute((0.5 - latent + eps))
        elif self.scoring_method == "reconstruction_loss":
            score = np.float64(avg_loss)
        elif self.scoring_method == "mix":
            latent_mean = np.absolute((0.5 - latent + eps))
            score = np.float64(
                (latent_mean * self.mix_weighting_factor)
                + (avg_loss / self.mix_weighting_factor)
            )
        else:
            raise ValueError(
                f"Unknown scoring_method: {self.scoring_method}! Choose 'latent_mean', 'reconstruction_loss' or 'mix'"
            )
        return score

    def _get_score_of_models_for_known_sequences(
        self, models: list[torch.nn.Module], x: torch.tensor
    ) -> np.array:
        scores = []
        if len(models) > 0:
            for cnt, model in enumerate(models):
                y_pred, latent = model(x)
                loss = self.loss_fn(y_pred, x, latent).cpu().detach().numpy()
                latent = latent[:, -1, :].cpu().detach().numpy()
                score = self._score_fn(loss, latent.mean(axis=1))
                if x.shape[0] == 1:
                    score = self.movingaverages[cnt].update(score)
                scores.append(score)
            scores_np = np.array(scores)
            return scores_np
        else:
            return None

    def _set_loss_function(
        self, loss_fun_name: str = "SparseRMSELoss"
    ) -> torch.nn.Module:
        if loss_fun_name == "RMSELoss":
            loss_fn = RMSELoss().to(device=self.device)
        elif loss_fun_name == "SparseRMSELoss":
            loss_fn = SparseRMSELoss(reduction="none").to(device=self.device)
        elif loss_fun_name == "SparseMSELoss":
            loss_fn = SparseMSELoss(reduction="none").to(device=self.device)
        return loss_fn

    def _set_model(self, ae_type: str):
        if ae_type == "GruAutoEncoder":
            model = GRUAutoEncoder(self.num_features, self.dim_bottleneck).to(
                self.device
            )
            self.rnn_model = True
        elif ae_type == "AutoEncoder":
            model = AutoEncoder().to(self.device)
            self.rnn_model = False
        elif ae_type == "BidirectionalGruAutoEncoder":
            model = BidirectionalGruAutoEncoder(
                dim_in=self.num_features,
                dim_bottleneck=self.dim_bottleneck,
                seed=self.seed,
            ).to(self.device)
            self.rnn_model = True
        else:
            raise NameError(
                "'ae_type' in config has a name for a model which is not defined"
            )
        if self.load_pretrained_base_model_from_file:
            try:
                model.load_state_dict(torch.load(self.pretrained_base_model_file_path))
            except FileNotFoundError as error:
                logger.error(
                    f"FileNotFoundError: Could not find file: {self.pretrained_base_model_file_path=}! Keep going with newly instantiated model.."
                )
        else:
            model = setWeightInitialization(
                model, use_weight_init=self.use_weight_init, seed=self.seed
            )
        return model

    def _set_seed(self):
        if self.seed is not None:
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)  # if you are using multi-GPU.
            np.random.seed(self.seed)  # Numpy module.
            random.seed(self.seed)  # Python random module.
            torch.manual_seed(self.seed)
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True)

    def _set_optimizer(self, optimizer_type: str = "Adam"):
        return set_optimizer(
            optimizer_type=optimizer_type,
            model_parameters=self.base_model.parameters(),
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            sgd_nesterov=True,
        )

    def _set_dim_bottleneck(self):
        if self.dim_bottleneck is None:
            # Automatically estimating the bottleneck dimension is not yet implemented. Get intrinsic dimensionality with PCA for example from the parameter set automatically function.
            logger.warning(f"dim_bottleneck is set to {self.num_features - 1}")
            return self.num_features - 1
        else:
            return self.dim_bottleneck

    def _adapt_learning_rate(self, new_lr):
        if self.adapt_learning_rate is True:
            for g in self.optimizer.param_groups:
                g["lr"] = new_lr

    def _buffer_training_results(
        self,
        x_orig: np.array,
        x_hat: np.array,
        latent: np.array,
        avg_loss: np.float32,
        step: int,
        losses_np: np.array,
        subs_id: int,
    ):
        new_row_list = [
            0,
            avg_loss,
            step * self.step_size,
            subs_id,
            len(self.list_models),
            latent.mean(),
        ]
        self.alldata_list.append(np.array(new_row_list))
        if losses_np is not None:
            for m, loss in enumerate(losses_np):
                new_row_list = [
                    m + 1,
                    loss,
                    step * self.step_size,
                    np.nan,
                    np.nan,
                    np.nan,
                ]
                self.alldata_list.append(np.array(new_row_list))
        if self.step_size > 1:
            x_hat = np.repeat(x_hat, self.step_size, axis=0)
        self.reconstruction_list.append(x_hat)
        return True

    def _save_buffered_results(self):
        if self.save_training_results is True:
            alldata_df = pd.DataFrame(
                data=np.stack(self.alldata_list), columns=self.alldata_columns
            )
            reconstruction_df = pd.DataFrame(
                data=np.concatenate(self.reconstruction_list)
            )
            reconstruction_df = reconstruction_df[: self._num_original_time_steps]
            Path(self.temp_folder).mkdir(parents=True, exist_ok=True)
            alldata_df.to_pickle(self.buffered_results_path)
            reconstruction_df.to_pickle(self.buffered_reconstruction_path)
            return True
        else:
            return False

    def _save_buffered_prediction_results(self):
        if self.save_training_results is True:
            column_names = [
                f"subs_id_{i + 1}" for i in range(self._prediction_losses.shape[1])
            ]
            pred_losses_df = pd.DataFrame(
                data=self._prediction_losses, columns=column_names
            )
            pred_losses_df.attrs["eta"] = self.eta
            pred_losses_df.attrs["theta"] = self.theta
            Path(self.temp_folder).mkdir(parents=True, exist_ok=True)
            pred_losses_df.to_pickle(self.buffered_prediction_results_path)
            return True
        else:
            return False

    def _drop_last_model(self):
        del self.list_models[-1]
        del self.lowest_subs_id_losses[-1]
        self.current_seq_counter -= 1

    def _compare_current_subs_loss_with_base_model_loss(
        self, subs_id: int, scores_np: np.array, score_ma: float
    ):
        """Take current subs_id loss and check if the previously lowest loss value is lower than the current subs_id loss. If it is higher, then the new lowest loss is set with the current subs_id loss.
        Take current base_model loss (score_ma) and compare if it is lower than any loss from the current subsequence before. Only if the base_model loss is lower than any previous loss from this subsequence model, then the parameters of the subsequence model are updated with the current base_model parameters.
        """
        if scores_np is not None:
            if (
                (len(scores_np) >= subs_id - 1)
                and (score_ma < scores_np[subs_id - 1])
                and (score_ma < self.lowest_subs_id_losses[subs_id - 1])
            ):
                # only do this the first time we are in this subsequence. If the subsequence appears multiple times in the training data, the model is not 'retrained'!
                if len(self.list_models) == (subs_id):
                    # Update current subsequence models parameters with base models parameters
                    self.lowest_subs_id_losses[subs_id - 1] = score_ma
                    self.list_models[subs_id - 1].load_state_dict(
                        self.base_model.state_dict()
                    )

    def _get_intrinsic_dimensionality(self, X: np.array, method: str = "pca") -> int:
        intrinsic_dimension = X.shape[1]
        if method == "pca":
            pca = PCA(n_components="mle")
            logger.info("Fitting the pca.. this can take a minute..")
            try:
                pca.fit(X)
                intrinsic_dimension = pca.n_components_
            except ValueError:
                logger.error(
                    "NUmber of components smaller than samples.. using components / 2"
                )
                intrinsic_dimension = int(intrinsic_dimension / 2)
            logger.info(
                f"Intrinsic dimension estimated from PCA :{intrinsic_dimension}"
            )
        else:
            raise NotImplementedError
        return intrinsic_dimension

    def _pre_data_analysis(self, X: np.array, dataset: Dataset) -> dict:
        # Find volatility:
        window_size = self.seq_len + self.phi
        volatility = pd.DataFrame(X).rolling(window_size).std().mean().mean()
        logger.debug(f"{volatility=}")
        window_sizes = np.linspace(2, 200, 100, dtype="int")
        vols = []
        for window in window_sizes:
            vols.append(pd.DataFrame(X).rolling(window).std().mean().mean())
        eta = np.log10(volatility)
        model = self._set_model(self.ae_type)
        loss_fn = self._set_loss_function(self.loss_function_name)
        lrf = LearningRateFinder(
            X,
            model,
            loss_fn,
            self.optimizer_type,
            self.device,
            self.seq_len,
            score_fn=self._score_fn,
            seed=self.seed,
        )
        best_lr, lowest_loss = lrf.find_lr()
        logger.debug(f"{best_lr=}")
        logger.debug(f"{lowest_loss=}")
        modemean = stats.mode(X)[0].mean()
        modestd = stats.mode(X)[0].std()
        logger.debug(f"{modemean=}")
        logger.debug(f"{modestd=}")
        meanmean = X.mean(axis=0).mean()
        logger.debug(f"{meanmean=}")
        meanstd = X.mean(axis=0).std()
        logger.debug(f"{meanstd=}")
        medianstd = np.median(X, axis=0).std()
        medianmean = np.median(X, axis=0).mean()
        logger.debug(f"{medianstd=}")
        logger.debug(f"{medianmean=}")
        ZXall = 0
        for i in range(X.shape[1]):
            fX, tX, ZXxx = signal.stft(X[:, i], fs=1)
            ZXall = np.abs(ZXxx) + ZXall
        stftmean = np.abs(ZXall).mean()
        logger.debug(f"{stftmean=}")
        intrinsic_dim = self._get_intrinsic_dimensionality(X)
        logger.debug(f"{intrinsic_dim=}")
        print(f"{intrinsic_dim=}")
        eta = lowest_loss * (1 - volatility) / (self.omega / 2)
        theta = lowest_loss * (1 - volatility)
        new_params = {
            "dim_bottleneck": intrinsic_dim,
            "learning_rate": best_lr,
            "eta": eta,
            "theta": theta,
            "theta_factor": theta / eta,
        }
        logger.debug(f"{new_params=}")
        return new_params


def _init_fn(worker_id):
    np.random.seed(42)
    random.seed(42)


torch_gen = torch.Generator()
torch_gen.manual_seed(42)


class MultipleTimeStepsDataset(Dataset):
    def __init__(
        self,
        X: np.array,
        y: np.array,
        seq_len: int,
        step_size: int = 1,
        device: str = "cpu",
        edge_padding_before=True,
        seed: int = 42,
    ):
        super().__init__()
        # Create unique instances from the provided instance labels
        instance_indexes = np.sort(np.unique(y, return_index=True)[1][1:])[1:]
        # Split the data according to the unique instance labels
        instances = np.split(X, instance_indexes)
        # The same with the labels
        labels = np.split(y, instance_indexes)
        # Instantiate empty lists for collecting the 'folded' data
        windowed_features = []
        windowed_labels = []
        # For loop for creating rolling time windows only of consecutive data points. Defined earlier by the instance labels.
        for instance, label in zip(instances, labels):
            if edge_padding_before is True:
                instance = np.pad(
                    instance, mode="edge", pad_width=((seq_len - 1, 0), (0, 0))
                )
                label = np.pad(label, mode="edge", pad_width=((seq_len - 1, 0)))
                # num_frames = instance.shape[0]
            num_frames = instance.shape[0] - seq_len + 1
            feature_windows = [
                instance[i : (i + seq_len)] for i in range(0, num_frames, step_size)
            ]
            feature_windows = np.stack(feature_windows)
            label_windows = [
                label[i : (i + seq_len)] for i in range(0, num_frames, step_size)
            ]
            label_windows = np.stack(label_windows)
            windowed_features.append(feature_windows)
            windowed_labels.append(label_windows)
        # Concatenate the rolling windows of each instance back to one matrix
        self.features = torch.tensor(
            np.concatenate(windowed_features), dtype=torch.float32
        )
        self.labels = torch.tensor(np.concatenate(windowed_labels), dtype=torch.float32)

    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        return (self.features[idx], self.labels[idx])

    def __len__(self) -> int:
        return len(self.features)


class MovingAverage:
    """Calculate moving average with ring buffer using the deque from collections"""

    def __init__(self, size=5):
        self._buf = deque(maxlen=size)

    def update(self, x):
        """Adds new value into the buffer"""
        self._buf.append(x)
        return np.array(self._buf).mean()

    def len(self):
        """Returns the length of the buffer"""
        return len(self._buf)


class LearningRateFinder:
    def __init__(
        self,
        X: np.array,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        optimizer_type: torch.optim.Optimizer,
        device: str,
        seq_len: int,
        score_fn: Callable[[float, float, None], np.float64],
        default_lr=1e-3,
        seed: int = 42,
    ) -> None:
        self.seed = seed
        self.torch_gen = None
        self._set_seed()
        self.X = X
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer_type = optimizer_type
        self.momentum = 0.9
        self.device = device
        self.lr_min = -6
        self.lr_max = 0
        self.num_epochs = 5
        self.batch_size = 1
        self.seq_len = seq_len
        self.num_samples = 100
        self.num_lr_samples = 15
        self.plot_lr_findings = False
        self.score_fn = score_fn
        self.default_lr = default_lr
        self.num_workers = 1
        self.pin_memory = False

    def _init_fn(self, worker_id):
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)  # if you are using multi-GPU.
        np.random.seed(self.seed)  # Numpy module.
        random.seed(self.seed)  # Python random module.
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)

    def _get_init_fn(self):
        if self.device != "cpu":
            return None
        else:
            return self._init_fn

    def _set_seed(self):
        if self.seed is not None:
            self.torch_gen = torch.Generator()
            self.torch_gen.manual_seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)  # if you are using multi-GPU.
            np.random.seed(self.seed)  # Numpy module.
            random.seed(self.seed)  # Python random module.
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.use_deterministic_algorithms(True)

    def pretrain(self, disable_progress_bar=False):
        X = self.X[: self.num_samples, :]
        y = np.full(X.shape[0], 0)
        # Create dataset with rolling windows of 'seq_len' length
        dataset = MultipleTimeStepsDataset(
            X, y, seq_len=self.seq_len, device=self.device, step_size=1
        )
        dataloader = DataLoader(
            dataset,
            self.batch_size,
            shuffle=False,
            pin_memory=self.pin_memory,
            collate_fn=None,
            num_workers=self.num_workers,
            worker_init_fn=self._get_init_fn(),
            generator=self.torch_gen,
        )
        with tqdm(
            total=len(dataset) * self.num_epochs, disable=disable_progress_bar
        ) as bar:
            epoch_avg_losses = []
            self.optimizer = set_optimizer(
                optimizer_type=self.optimizer_type,
                model_parameters=self.model.parameters(),
                learning_rate=self.default_lr,
                momentum=self.momentum,
                sgd_nesterov=True,
            )
            batch_losses = []
            for epoch in range(self.num_epochs):
                for step, data in enumerate(dataloader):
                    x = data[0].to(self.device)
                    y_pred, latent = self.model(x)
                    loss = self.loss_fn(y_pred, x, latent)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    d_loss = loss.cpu().detach().numpy().item()
                    score = self.score_fn(d_loss, latent.mean().cpu().detach().numpy())
                    batch_losses.append(score)
                    bar.update()
                    bar.set_description(f"Pretraining.. score = {score:.3e}")
                epoch_avg_losses.append(np.array(batch_losses).mean())
        return self.model

    def find_lr(self):
        rng = np.random.default_rng()
        X = rng.choice(self.X, self.num_samples)
        y = np.full(X.shape[0], 0)
        # Create dataset with rolling windows of 'seq_len' length
        dataset = MultipleTimeStepsDataset(
            X, y, seq_len=self.seq_len, device=self.device, step_size=1
        )
        dataloader = DataLoader(
            dataset,
            self.batch_size,
            shuffle=False,
            pin_memory=False,
            collate_fn=None,
            num_workers=0,
            worker_init_fn=_init_fn,
            generator=torch_gen,
        )
        lrs = np.logspace(self.lr_min, self.lr_max, num=self.num_lr_samples)
        with tqdm(
            total=len(dataset) * len(lrs) * self.num_epochs, disable=False
        ) as bar:
            loss_lrs = []
            for lr in lrs:
                epoch_avg_losses = []
                self.optimizer = set_optimizer(
                    optimizer_type=self.optimizer_type,
                    model_parameters=self.model.parameters(),
                    learning_rate=lr,
                    momentum=self.momentum,
                    sgd_nesterov=True,
                )
                batch_losses = []
                for epoch in range(self.num_epochs):
                    for step, data in enumerate(dataloader):
                        x = data[0].to(self.device)
                        y_pred, latent = self.model(x)
                        loss = self.loss_fn(y_pred, x, latent)
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        d_loss = loss.cpu().detach().numpy().item()
                        score = self.score_fn(
                            d_loss, latent.mean().cpu().detach().numpy()
                        )
                        batch_losses.append(score)
                        bar.update()
                        bar.set_description(f"Finding Learning rate.. {lr:.2e}")
                    epoch_avg_losses.append(np.array(batch_losses).mean())
                loss_lrs.append(np.array(epoch_avg_losses).mean())
            losses = np.array(loss_lrs)
            interp_lrs = np.logspace(self.lr_min, self.lr_max, num=100)
            losses_interp = np.interp(interp_lrs, lrs, losses)
            cumsum_vec = np.cumsum(np.insert(losses_interp, 0, 0))
            window_width = 5
            losses_i = (
                cumsum_vec[window_width:] - cumsum_vec[:-window_width]
            ) / window_width
            lograd_i = np.gradient(losses_i)
            best_idx = lograd_i.argmin()
            if self.plot_lr_findings is True:
                fig, ax = plt.subplots()
                ax.set_title(f"Best learning rate found: {interp_lrs[best_idx]:.2e}")
                ax.plot(interp_lrs[: len(losses_i)], losses_i, label="loss")
                ax.scatter(
                    interp_lrs[best_idx],
                    losses_i[best_idx],
                    color="r",
                    alpha=0.9,
                    label="max negativ gradient",
                )
                ax.legend()
                ax.set_xlabel("Learning Rate")
                ax.set_ylabel("Loss")
                plt.xscale("log")
                plt.tight_layout()
                plt.savefig("lr_finder.png")
                plt.close()
        best_lr = interp_lrs[best_idx]
        lowest_loss = losses_i.min()
        return best_lr, lowest_loss


if __name__ == "__main__":
    pass
else:
    # Standard Libraries
    import logging

    logger = logging.getLogger(LOGGER_NAME)
