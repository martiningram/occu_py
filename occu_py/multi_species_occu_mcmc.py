from .checklist_model import ChecklistModel
import numpy as np
import pandas as pd
from .functional.hierarchical_checklist_model_mcmc import fit, predict_env, predict_obs
from os import makedirs
from ml_tools.utils import save_pickle_safely, load_pickle_safely
from os.path import join
from ml_tools.patsy import save_design_info, restore_design_info
import arviz as az


class MultiSpeciesOccuMCMC(ChecklistModel):
    def __init__(
        self,
        env_formula,
        obs_formula,
        n_draws=1000,
        n_tune=1000,
        thinning=1,
        chain_method="vectorized",
    ):

        self.scaler = None
        self.env_formula = env_formula
        self.obs_formula = obs_formula
        self.n_draws = n_draws
        self.n_tune = n_tune
        self.thinning = thinning
        self.chain_method = chain_method

    def fit(
        self,
        X_env: pd.DataFrame,
        X_checklist: pd.DataFrame,
        y_checklist: pd.DataFrame,
        checklist_cell_ids: np.ndarray,
    ):

        self.X_env = X_env
        self.X_checklist = X_checklist

        self.samples, self.design_info = fit(
            X_env,
            X_checklist,
            y_checklist,
            checklist_cell_ids,
            self.env_formula,
            self.obs_formula,
            scale_env=False,
            draws=self.n_draws,
            tune=self.n_tune,
            thinning=self.thinning,
            chain_method=self.chain_method,
        )

    def predict_marginal_probabilities_direct(self, X: pd.DataFrame) -> pd.DataFrame:

        return predict_env(X, self.samples, self.design_info)

    def predict_marginal_probabilities_obs(
        self, X: pd.DataFrame, X_obs: pd.DataFrame
    ) -> pd.DataFrame:

        return predict_obs(X, X_obs, self.samples, self.design_info)

    def save_model(self, target_folder: str) -> None:
        # TODO: Test this

        makedirs(target_folder, exist_ok=True)
        save_pickle_safely(
            {
                "env_coef_names": self.design_info["env"].column_names,
                "obs_coef_names": self.design_info["obs"].column_names,
                "env_scaler": None
                if "env_scaler" not in self.design_info
                else self.design_info["env_scaler"],
                "species_names": self.design_info["species_names"],
                "env_formula": self.env_formula,
                "obs_formula": self.obs_formula,
            },
            join(target_folder, "design_info.pkl"),
        )
        self.samples.to_netcdf(join(target_folder, "mcmc_samples.netcdf"))

        # Save the design infos
        save_design_info(
            self.X_env,
            self.env_formula,
            self.design_info["env"],
            join(target_folder, "design_info_env.pkl"),
        )

        save_design_info(
            self.X_checklist,
            self.obs_formula,
            self.design_info["obs"],
            join(target_folder, "design_info_obs.pkl"),
        )

    def restore_model(self, restore_folder: str) -> None:
        # TODO: Test this

        self.samples = az.from_netcdf(join(restore_folder, "mcmc_samples.netcdf"))

        env_design_info = restore_design_info(
            join(restore_folder, "design_info_env.pkl")
        )

        obs_design_info = restore_design_info(
            join(restore_folder, "design_info_obs.pkl")
        )

        other_design_info = load_pickle_safely(join(restore_folder, "design_info.pkl"))

        self.env_formula = other_design_info["env_formula"]
        self.obs_formula = other_design_info["obs_formula"]

        self.design_info = {
            "env": env_design_info,
            "obs": obs_design_info,
            "species_names": other_design_info["species_names"],
        }
