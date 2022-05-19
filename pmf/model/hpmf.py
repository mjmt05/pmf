"""Model class for heirarchical poisson matrix factorization."""
import logging
import numpy as np
from scipy.stats import poisson

logger = logging.getLogger(__name__)


class HPMF:
    """HPMF stores the parameters and methods for the heirarchical Poisson
    matrix factorization model.
    TODO: Specify paper"""

    def __init__(
        self, nusers, nitems, latent_dimensions=3, bernoulli_poisson=True, **kwargs
    ):
        """The constucter for the HPMF class.

        Arguments:
            nusers (int): Total number of users.
            nitems (int): Total number of items.
            latent_dimensions (int): Number of latent dimensions.
            bernoulli_poisson (bool): Use Bernoulli-Poisson version of the model.

        Keyword Arguments:
            alpha_shape_prior (float): Shape parameter for the gamma prior for the user latent
                                       parameters. Default 1.0.
            beta_shape_prior (float): Shape parameter for the gamma prior for the item latent
                                      parameters. Default 1.0.
            zeta_alpha_shape (float): Shape parameter for the gamma hyperprior for alpha.
                                      Default 1.0.
            zeta_beta_shape (float): Shape parameter for the gamma hyperprior for beta.
                                     Default 1.0.
            zeta_alpha_rate (float): Rate parameter for the gamma hyperprior for alpha.
                                     Default 0.1.
            zeta_beta_rate (float): Rate parameter for the gamma hyperprior for beta.
                                    Default 0.1.
        """
        self.alpha_shape_prior = kwargs.get("alpha_shape_prior", 1.0)
        self.alpha = None
        self.beta_shape_prior = kwargs.get("beta_shape_prior", 1.0)
        self.beta = None
        self.berpo = bernoulli_poisson
        self.zeta_alpha_shape_prior = kwargs.get("zeta_alpha_shape", 1.0)
        self.zeta_beta_shape_prior = kwargs.get("zeta_beta_shape", 1.0)
        self.zeta_alpha_rate_prior = kwargs.get("zeta_alpha_rate", 0.1)
        self.zeta_beta_rate_prior = kwargs.get("zeta_beta_rate", 0.1)
        self.zeta_alpha = None
        self.zeta_beta = None
        self.latent_dimensions = latent_dimensions
        self.nusers = nusers
        self.nitems = nitems

    def pmf(self, user=None, item=None, count=None):
        """Return probability mass function for user, item and count.
        If user and item not given returns a matrix with the pmf for the matrix of counts.
        If count not given sets count to one.

        Arguments:
        user (int): Integer id for user.
        item (int): Integer id for item.
        count (int): Count for user and item.
        """
        if user is not None and not isinstance(user, int):
            raise ValueError("User must be of type int.")

        if item is not None and not isinstance(item, int):
            raise ValueError("Item must be of type int.")

        if user is not None and item is not None:
            poisson_rate = sum(self.alpha[user] * self.beta[item])
        else:
            poisson_rate = self.alpha @ self.beta.T

        if self.berpo:
            return 1 - np.exp(-poisson_rate)

        if count is None:
            count = 1
        return poisson.pmf(count, poisson_rate)

    def _simulate_alpha(self):
        self.zeta_alpha = np.random.gamma(
            self.zeta_alpha_shape_prior, 1.0 / self.zeta_alpha_rate_prior, self.nusers
        )
        self.alpha = np.full((self.nusers, self.latent_dimensions), 0.0)
        for i in range(self.nusers):
            self.alpha[i, :] = np.random.gamma(
                self.alpha_shape_prior, 1.0 / self.zeta_alpha[i], self.latent_dimensions
            )

    def _simulate_beta(self):
        self.zeta_beta = np.random.gamma(
            self.zeta_beta_shape_prior, 1.0 / self.zeta_beta_rate_prior, self.nitems
        )
        self.beta = np.full((self.nitems, self.latent_dimensions), 0.0)
        for i in range(self.nusers):
            self.beta[i, :] = np.random.gamma(
                self.beta_shape_prior, 1.0 / self.zeta_beta[i], self.latent_dimensions
            )

    def simulate_model_parameters(self):
        """Randomly sample parameters from the prior distributions."""
        self._simulate_alpha()
        self._simulate_beta()

    def simulate_counts(self):
        """Simulate counts from the prior.

        Returns: A numpy matrix of size (nusers, nitems) with counts."""
        if self.alpha is None:
            self._simulate_alpha()
        if self.beta is None:
            self._simulate_beta()
        poisson_rate = self.alpha @ self.beta.T
        counts = np.random.poisson(poisson_rate)
        if self.berpo:
            return np.matrix(np.matrix(counts, dtype=bool), dtype=int)
        return counts

    def write_model_state(self, outpath, user_maps=None, item_maps=None):
        """Write the user and item latent parameters to outpath with schema
        user/item,latent_param1,...,latent_paramk.

        Arguments:
            outpath (str): Path to write model parameters alpha.txt
                           and beta.txt.
            user_maps (dict): Mappings for the integer user ids if it exists.
            item_maps (dict): Mappings for the integer item ids if it exists.
        """
        if self.alpha is None:
            logger.warning(
                "Can not write user latent parameters to file as it has not been set."
            )
        else:
            with open(outpath + "alpha.txt", "w", encoding="utf-8") as falpha:
                for idx, row in enumerate(self.alpha):
                    if user_maps:
                        user = user_maps[idx]
                    else:
                        user = str(idx)
                    print(",".join([user] + [str(i) for i in row]), file=falpha)

        if self.beta is None:
            logger.warning(
                "Can not write item latent parameters to file as it has not been set."
            )
        else:
            with open(outpath + "beta.txt", "w", encoding="utf-8") as fbeta:
                for idx, row in enumerate(self.beta):
                    if item_maps:
                        item = item_maps[idx]
                    else:
                        item = str(idx)
                    print(",".join([item] + [str(i) for i in row]), file=fbeta)
