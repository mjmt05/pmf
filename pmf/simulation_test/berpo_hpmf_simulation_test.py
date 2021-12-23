#! /usr/bin/env python
"""Simulate a train and test data set based on the Bernoulli-Poisson HPMF
model and run hpmf inference algorithm on training data. If the algorithm
is working correctly and the parameters are specified such that the graph
is sparse (number of training edges to total possible edges) the AUC for
the test data should be high."""
import sys
import logging
import itertools
import argparse
import numpy as np
from sklearn.metrics import roc_auc_score
from pmf.model.hpmf import HPMF
from pmf.data.datastore import Data
from pmf.vi.vi import VI


def create_edgelist_from_counts(counts):
    """Return a list of tuples containing the edge indices and counts
    from a matrix of counts."""
    return [(i, j, counts[i, j]) for i, j in zip(*np.where(counts >= 1))]


def run_simulation(args):
    """Run simulation."""

    if args.seed is not None:
        np.random.seed(args.seed)
    hpmf = HPMF(
        args.nusers,
        args.nitems,
        latent_dimensions=args.latent_dimensions,
        zeta_alpha_shape=args.zeta_alpha_shape,
        zeta_alpha_rate=args.zeta_alpha_rate,
        zeta_beta_shape=args.zeta_beta_shape,
        zeta_beta_rate=args.zeta_beta_rate,
        alpha_shape_prior=args.alpha_shape,
        beta_rate_prior=args.beta_shape,
    )
    hpmf.simulate_model_parameters()
    # Create train and test set.
    train = hpmf.simulate_counts()
    possible_edges = args.nusers * args.nitems
    print(f"Sparsity of training data: {np.sum(train > 0) / possible_edges}")
    test = hpmf.simulate_counts()
    print(f"Sparsity of test data: {np.sum(test > 0) / possible_edges}")

    # Create edgelist for training data.
    dataobj = Data(
        edgelist=create_edgelist_from_counts(train),
        userlist=list(range(args.nusers)),
        itemlist=list(range(args.nitems)),
    )
    # Run inference on training data.
    inference = VI(dataobj, hpmf, seed=args.seed)
    inference.run_algorithm()

    label = []
    score = []
    for i, j in itertools.product(range(args.nusers), range(args.nitems)):
        user = dataobj.get_id_for_user(i)
        item = dataobj.get_id_for_item(j)
        label.append(test[user, item])
        score.append(hpmf.pmf(user, item, 1))
    print("AUC score for test data: ", roc_auc_score(label, score))


def main():
    """Parses command-line arguments and runs simulation."""
    formatter = argparse.ArgumentDefaultsHelpFormatter
    commandlineargs = argparse.ArgumentParser(formatter_class=formatter)
    commandlineargs.add_argument(
        "-u",
        "--nusers",
        default=500,
        type=int,
        dest="nusers",
        help="Number of users for simulation.",
    )
    commandlineargs.add_argument(
        "-i",
        "--nitems",
        default=500,
        type=int,
        dest="nitems",
        help="Number of items for simulation.",
    )
    commandlineargs.add_argument(
        "-l",
        "--latent-dimensions",
        type=int,
        dest="latent_dimensions",
        default=3,
        help="Number of latent factors.",
    )
    commandlineargs.add_argument(
        "-a",
        "--alpha-shape",
        type=float,
        dest="alpha_shape",
        default=1.0,
        help="Shape parameter for the gamma prior for alpha.",
    )
    commandlineargs.add_argument(
        "-b",
        "--beta-shape",
        type=float,
        dest="beta_shape",
        default=1.0,
        help="Shape parameter for the gamma prior for beta.",
    )
    commandlineargs.add_argument(
        "-za",
        "--zeta-alpha-shape",
        type=float,
        dest="zeta_alpha_shape",
        default=1.0,
        help="Shape parameter for the gamma hyperprior for alpha.",
    )
    commandlineargs.add_argument(
        "-zb",
        "--zeta-beta-shape",
        type=float,
        dest="zeta_beta_shape",
        default=1.0,
        help="Shape parameter for the gamma hyperprior for beta.",
    )
    commandlineargs.add_argument(
        "-ca",
        "--zeta-alpha-rate",
        type=float,
        dest="zeta_alpha_rate",
        default=0.05,
        help="Rate parameter for the gamma hyperprior for alpha.",
    )
    commandlineargs.add_argument(
        "-cb",
        "--zeta-beta-rate",
        type=float,
        dest="zeta_beta_rate",
        default=0.05,
        help="Rate parameter for the gamma hyperprior for beta.",
    )
    commandlineargs.add_argument(
        "-s",
        "--seed",
        type=int,
        dest="seed",
        help="Seed numpy random number generator with %(dest)s for consistent results.",
        default=None,
    )
    args = commandlineargs.parse_args()
    logging.basicConfig(level=logging.INFO)
    run_simulation(args)


if __name__ == "__main__":
    sys.exit(main())
