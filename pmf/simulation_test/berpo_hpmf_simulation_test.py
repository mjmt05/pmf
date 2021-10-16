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
    nusers = 10
    nitems = 10
    latent_dimensions = 5
    seed = 1
    np.random.seed(seed)
    hpmf = HPMF(10, 10, latent_dimensions=latent_dimensions)
    hpmf.simulate_model_parameters()
    train = hpmf.simulate_counts()
    test = hpmf.simulate_counts()
    dataobj = Data(edgelist=create_edgelist_from_counts(train))
    inference = VI(dataobj, hpmf, seed=seed)
    inference.run_algorithm()
    label = []
    score = []
    for i, j in itertools.product(range(nusers), range(nitems)):
        label.append(test[i, j])
        score.append(hpmf.pmf(i, j, 1))
    print("AUC score for test data: ", roc_auc_score(label, score))


if __name__ == "__main__":
    sys.exit(main())
