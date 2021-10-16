"""Model class for heirarchical poisson matrix factorization."""
import sys
class HPMF:
    """HPMF stores the parameters and methods for the heirarchical Poisson
    matrix factorization model.
    TODO: Specify paper"""

    def __init__(
        self,
        nusers,
        nitems,
        latent_dimensions=3,
        bernoulli_poisson=True,
        **kwargs
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
        self.latent_dimensions = latent_dimensions
        self.nusers = nusers
        self.nitems = nitems

    def pmf(self, user, item, count):
        """Return probability mass function for user, item and count.

        Arguments:
        user (int): Id for user.
        item (int): Id for item.
        count (int): Count for user and item.
        """

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
            print("Alpha has not been set.", file=sys.stderr)
        else:
            with open(outpath + "alpha.txt", "w", encoding="utf-8") as falpha:
                for idx, row in enumerate(self.alpha):
                    if user_maps:
                        user = user_maps[idx]
                    else:
                        user = str(idx)
                    print(",".join([user] + [str(i) for i in row]), file=falpha)

        if self.beta is None:
            with open(outpath + "beta.txt", "w", encoding="utf-8") as fbeta:
                for idx, row in enumerate(self.beta):
                    if item_maps:
                        item = item_maps[idx]
                    else:
                        item = str(idx)
                    print(",".join([item] + [str(i) for i in row]), file=fbeta)
