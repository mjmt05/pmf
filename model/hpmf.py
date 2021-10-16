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
        alpha_shape_prior=1,
        beta_shape_prior=1,
    ):
        """The constucter for the HPMF class.

        Arguments:
            nusers (int): Total number of users.
            nitems (int): Total number of items.
            latent_dimensions(int): Number of latent dimensions.
            alpha_shape_prior (float): Prior for the user latent parameters.
            beta_shape_prior (float): Prior for the item latent parameters.
        """
        self.alpha_shape_prior = alpha_shape_prior
        self.alpha = None
        self.beta_shape_prior = beta_shape_prior
        self.beta = None
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
