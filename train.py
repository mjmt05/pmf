#! /usr/bin/env python3
import sys
import argparse
from pmf.data.datastore import Data
from pmf.model.hpmf import HPMF
from pmf.vi.vi import VI


def main():
    """Parses command-line arguments and reads in data and executes inference algorithm."""
    formatter = argparse.ArgumentDefaultsHelpFormatter
    commandlineargs = argparse.ArgumentParser(formatter_class=formatter)
    commandlineargs.add_argument(
        "-f",
        "--edgelist-path",
        type=str,
        dest="edgelist",
        required=True,
        help="Load training edge list from %(dest)s",
    )
    commandlineargs.add_argument(
        "-l",
        "--latent-factors",
        type=int,
        dest="latent_factors",
        default=5,
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
        default=0.1,
        help="Rate parameter for the gamma hyperprior for alpha.",
    )
    commandlineargs.add_argument(
        "-cb",
        "--zeta-beta-rate",
        type=float,
        dest="zeta_beta_rate",
        default=0.1,
        help="Rate parameter for the gamma hyperprior for beta.",
    )
    commandlineargs.add_argument(
        "-p",
        "--bernoulli-poisson",
        dest="berpo",
        action="store_true",
        default=False,
        help="Use Bernoulli-Poisson model (treat counts as binary).",
    )
    commandlineargs.add_argument(
        "-t",
        "--threshold",
        type=float,
        dest="convergence_criterion",
        default=0.00001,
        help="Convergence threshold based on relative difference\
                                 between two consecutive values of the ELBO.",
    )
    commandlineargs.add_argument(
        "-m",
        "--max-iterations",
        type=int,
        dest="max_iterations",
        default=500,
        help="Max iterations for the variational inference algorithm.",
    )
    commandlineargs.add_argument(
        "-o",
        "--output-dir",
        type=str,
        dest="output_directory",
        default="",
        help="Output latent parameters to %(dest)s",
    )
    commandlineargs.add_argument(
        "-s",
        "--seed",
        type=int,
        dest="seed",
        help="Seed numpy random number generator with %(dest)s for consistent results.",
    )
    args = commandlineargs.parse_args()
    data = Data(edgelist_path=args.edgelist)
    hpmf = HPMF(
        data.get_number_users(),
        data.get_number_items(),
        latent_dimensions=args.latent_factors,
        bernoulli_poisson=args.berpo,
        alpha_shape_prior= args.alpha_shape,
        beta_shape_prior= args.beta_shape,
        zeta_alpha_shape= args.zeta_alpha_shape,
        zeta_beta_shape= args.zeta_beta_shape,
        zeta_alpha_rate= args.zeta_alpha_rate,
        zeta_beta_rate= args.zeta_beta_rate,
    )
    inference = VI(data, hpmf, args.seed, args.convergence_criterion, args.max_iterations)
    inference.run_algorithm()
    hpmf.write_model_state(
        args.output_directory, data.get_user_id_mappings(), data.get_item_id_mappings()
    )


if __name__ == "__main__":
    sys.exit(main())
