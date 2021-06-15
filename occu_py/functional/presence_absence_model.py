import numpy as np
import jax.numpy as jnp
from patsy import dmatrix, build_design_matrices
from ml_tools.patsy import remove_intercept_column
from jax.nn import log_sigmoid
from jax.scipy.stats import norm
from jax_advi.advi import optimize_advi_mean_field
from functools import partial
from scipy.special import expit


def compute_likelihood(theta, y, X):

    env_logits = X @ theta["beta_env"] + theta["intercept"]

    log_prob_pres = log_sigmoid(env_logits)
    log_prob_abs = log_sigmoid(-env_logits)

    lik_terms = y * log_prob_pres + (1 - y) * log_prob_abs

    return jnp.sum(lik_terms)


def compute_prior(theta, prior_sd, intercept_prior_sd):

    return jnp.sum(norm.logpdf(theta["beta_env"], 0.0, prior_sd)) + norm.logpdf(
        theta["intercept"], 0.0, intercept_prior_sd
    )


def fit(formula, X_df, y, prior_sd=1.0, intercept_prior_sd=10.0, M=100, verbose=False):

    design_mat = dmatrix(formula, X_df)
    X_env_mat = np.asarray(design_mat)
    X_env_mat = remove_intercept_column(X_env_mat, design_mat.design_info)

    y = np.array(y).astype(float)

    theta_shape_dict = {"beta_env": (X_env_mat.shape[1],), "intercept": ()}
    cur_lik = partial(compute_likelihood, y=y, X=X_env_mat)
    cur_prior = partial(compute_prior, prior_sd=prior_sd)

    result = optimize_advi_mean_field(
        theta_shape_dict, cur_prior, cur_lik, verbose=verbose, M=M
    )

    return {
        "draws": result["draws"],
        # "opt_info": result,
        "design_info": design_mat.design_info,
    }


def predict(X_new, draws, design_info):

    beta_draws = draws["beta_env"]
    int_draws = draws["intercept"]

    env_design_mat = np.asarray(build_design_matrices([design_info], X_new)[0])
    env_design_mat = remove_intercept_column(env_design_mat, design_info)

    probs = expit(
        np.einsum("ij,kj->ki", env_design_mat, beta_draws) + int_draws.reshape(-1, 1)
    ).mean(axis=0)

    return probs
