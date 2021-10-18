#! /usr/bin/env python
"""Simulate data and run algorithm."""
import sys
import itertools
import numpy as np
from sklearn.metrics import roc_auc_score
from pmf.model.hpmf import HPMF
from pmf.data.datastore import Data
from pmf.vi.vi import VI


def create_edgelist_from_counts(counts):
    return [(i, j, counts[i, j]) for i, j in zip(*np.where(counts >= 1))]


def main():
    """Parse command line arguments and exit."""
    nusers = 500
    nitems = 500
    latent_dimensions = 3
    seed = 5
    zeta_shape = 1.0
    zeta_rate = 0.05
    latent_shape = 1.0
    # Want to decrease average while increasing variance
    np.random.seed(seed)
    hpmf = HPMF(
        nusers,
        nitems,
        latent_dimensions=latent_dimensions,
        zeta_alpha_shape=zeta_shape,
        zeta_alpha_rate=zeta_rate,
        zeta_beta_shape=zeta_shape,
        zeta_beta_rate=zeta_rate,
        alpha_shape_prior=latent_shape,
        beta_rate_prior=latent_shape,
    )
    hpmf.simulate_model_parameters()
    train = hpmf.simulate_counts()
    possible_edges = nusers * nitems
    total_training_edges = np.sum(train > 0)
    print(f"Sparsity of training data: {total_training_edges / possible_edges}")
    test = hpmf.simulate_counts()
    total_test_edges = np.sum(train > 0)
    print(f"Sparsity of test data: {total_test_edges / possible_edges}")

    dataobj = Data(
        edgelist=create_edgelist_from_counts(train),
        userlist=list(range(nusers)),
        itemlist=list(range(nitems)),
    )
    # Recreate the model object as some users or items there will be no observed edges.
    inference = VI(dataobj, hpmf, seed=seed)
    inference.run_algorithm()
    label = []
    score = []
    for i, j in itertools.product(range(nusers), range(nitems)):
        user = dataobj.get_id_for_user(i)
        item = dataobj.get_id_for_item(j)
        label.append(test[user, item])
        score.append(hpmf.pmf(user, item, 1))
    print("AUC score for test data: ", roc_auc_score(label, score))


if __name__ == "__main__":
    sys.exit(main())
