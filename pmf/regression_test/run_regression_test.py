#! /usr/bin/env python3
"""The module runs the vi inference algorithm over the edgelist given by
train.txt and checks the parameters alpha and beta against the alpha_gold
and beta_gold to ensure that they are the same."""
import sys
import logging
import numpy as np
from pmf.data.datastore import Data
from pmf.model.hpmf import HPMF
from pmf.vi.vi import VI


def run_vi_algorithm(berpo, data):
    """Runs the variational inference algorithm and returns the latent
    parameters alpha and beta."""
    seed = 1
    hpmf = HPMF(
        data.get_number_users(),
        data.get_number_items(),
        bernoulli_poisson=berpo,
        latent_dimensions=5,
    )
    inference = VI(data, hpmf, seed)
    inference.run_algorithm()
    return hpmf.alpha, hpmf.beta


def main():
    """Runs regression test for both bernoulli poisson and regular model,
    defaults for model and algorithm are used."""
    logging.basicConfig(level=logging.INFO)
    data = Data(edgelist_path="train.txt")
    # Run regression test with regular model.
    alpha, beta = run_vi_algorithm(False, data)
    alpha_gold = np.load("alpha_gold.npy")
    beta_gold = np.load("beta_gold.npy")
    np.testing.assert_array_almost_equal(alpha, alpha_gold, decimal=10)
    np.testing.assert_array_almost_equal(beta, beta_gold, decimal=10)
    #Run regression test with Bernoulli-Poisson model
    alpha, beta = run_vi_algorithm(True, data)
    alpha_gold = np.load("alpha_gold_bp.npy")
    beta_gold = np.load("beta_gold_bp.npy")
    np.testing.assert_array_almost_equal(alpha, alpha_gold, decimal=10)
    np.testing.assert_array_almost_equal(beta, beta_gold, decimal=10)
    logging.info("All regression tests passed.")

if __name__ == "__main__":
    sys.exit(main())
