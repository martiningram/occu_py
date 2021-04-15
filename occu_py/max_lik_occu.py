from .checklist_model import ChecklistModel
import numpy as np
import pandas as pd
import jax.numpy as jnp
from .likelihoods import compute_checklist_likelihood
from functools import partial
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from jax.nn import log_sigmoid, sigmoid
import os
from os.path import join
from typing import Callable
import pickle
from .functional.max_lik_occu_model import fit, predict_env_logit, predict_obs_logit
from ml_tools.patsy import create_formula, save_design_info, restore_design_info
from glob import glob
import jax


class MaxLikOccu(ChecklistModel):
    def __init__(self, env_formula, det_formula):

        self.fit_results = None
        self.species_names = None
        self.env_formula = env_formula
        self.det_formula = det_formula

    def fit(
        self,
        X_env: pd.DataFrame,
        X_checklist: pd.DataFrame,
        y_checklist: pd.DataFrame,
        checklist_cell_ids: np.ndarray,
    ) -> None:

        self.X_env = X_env
        self.X_checklist = X_checklist

        self.fit_results = list()
        self.species_names = y_checklist.columns

        for cur_species in tqdm(self.species_names):

            cur_y_checklist = y_checklist[cur_species].values

            fit_result = fit(
                X_env,
                X_checklist,
                cur_y_checklist,
                checklist_cell_ids,
                self.env_formula,
                self.det_formula,
                scale_env_data=False,
            )

            # Make sure JAX clears its memory:
            backend = jax.lib.xla_bridge.get_backend()
            for buf in backend.live_buffers():
                buf.delete()

            # print(
            #     cur_species,
            #     fit_result["optimisation_successful"],
            #     np.linalg.norm(fit_result["opt_result"].jac),
            # )

            self.fit_results.append(fit_result)

        self.env_design_info = self.fit_results[0]["env_design_info"]
        self.obs_design_info = self.fit_results[0]["obs_design_info"]

    def predict_marginal_probabilities_direct(self, X: pd.DataFrame) -> np.ndarray:

        predictions = list()

        for cur_species_name, cur_fit_result in zip(
            self.species_names, self.fit_results
        ):

            cur_prediction = predict_env_logit(
                X, self.env_design_info, cur_fit_result["env_coefs"]
            )

            cur_prob_pres = sigmoid(cur_prediction)
            predictions.append(cur_prob_pres)

        predictions = np.stack(predictions, axis=1)

        return pd.DataFrame(predictions, columns=self.species_names)

    def predict_marginal_probabilities_obs(self, X: pd.DataFrame, X_obs: pd.DataFrame):

        predictions = list()

        for cur_species_name, cur_fit_result in zip(
            self.species_names, self.fit_results
        ):

            cur_env_prediction = predict_env_logit(
                X, self.env_design_info, cur_fit_result["env_coefs"]
            )

            cur_obs_prediction = predict_obs_logit(
                X_obs, self.obs_design_info, cur_fit_result["obs_coefs"]
            )

            cur_log_prob_obs = log_sigmoid(cur_env_prediction) + log_sigmoid(
                cur_obs_prediction
            )
            predictions.append(np.exp(cur_log_prob_obs))

        predictions = np.stack(predictions, axis=1)

        return pd.DataFrame(predictions, columns=self.species_names)

    def save_model(self, target_folder: str) -> None:

        os.makedirs(target_folder, exist_ok=True)

        # Save the results files
        for i, (cur_results, cur_species) in enumerate(
            zip(self.fit_results, self.species_names)
        ):
            np.savez(
                os.path.join(target_folder, f"results_file_{i}"),
                species_name=cur_species,
                env_coefs=cur_results["env_coefs"],
                obs_coefs=cur_results["obs_coefs"],
                successful=cur_results["optimisation_successful"],
                env_formula=self.env_formula,
                det_formula=self.det_formula,
                final_grad_norm=np.linalg.norm(cur_results["opt_result"].jac),
            )

        # Save the design infos
        save_design_info(
            self.X_env,
            self.env_formula,
            self.env_design_info,
            join(target_folder, "design_info_env.pkl"),
        )

        save_design_info(
            self.X_checklist,
            self.det_formula,
            self.obs_design_info,
            join(target_folder, "design_info_obs.pkl"),
        )

    def restore_model(self, load_folder: str) -> None:

        all_results_files = glob(join(load_folder, "*.npz"))

        # Make sure these are sorted:
        def find_number(cur_file):

            filename = os.path.splitext(os.path.split(cur_file)[-1])[0]
            number = int(filename.split("_")[-1])

            return number

        sorted_files = sorted(all_results_files, key=find_number)

        loaded = [np.load(x) for x in sorted_files]

        self.env_formula = loaded[0]["env_formula"]
        self.det_formula = loaded[0]["det_formula"]

        self.env_design_info = restore_design_info(
            join(load_folder, "design_info_env.pkl")
        )

        self.obs_design_info = restore_design_info(
            join(load_folder, "design_info_obs.pkl")
        )

        self.fit_results = loaded
        self.species_names = [str(x["species_name"]) for x in loaded]
