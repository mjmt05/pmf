#! /usr/bin/env python
"""This module performs variational inference on the pmf model."""
import sys
import numpy as np
from scipy.special import digamma, gammaln


class VI:
    """VI stores the variational distribution parameters and
    contains all the methods to perform inference."""

    def __init__(
        self, data, model, seed=None, convergence_threshold=0.00001, max_iterations=500
    ):
        """The constructor for the VI class.

        Arguments:
        data (object): The data class which contains the edgelist.
        model (object): The model class object.
        seed (int): Seed the numpy random number generator. This will guarantee identical results
                    each run.
        convergence_threshold (float): Threshold for convergence based on the relative difference
                                       between two consecutive values of the ELBO.
        max_iterations (int): Max iterations to allow fo the variational inference algorithm.
                              Algorithm will exit when either convergence threshold or max
                              iterations is reached.
        """
        if seed is not None:
            np.random.seed(seed)
        self._model = model
        self._lambda_user = self._model.alpha_shape_prior * np.ones(
            (self._model.nusers, self._model.latent_dimensions)
        )
        self._digamma_lambda_user = digamma(self._lambda_user)
        self._lambda_item = self._model.beta_shape_prior * np.ones(
            (self._model.nitems, self._model.latent_dimensions)
        )
        self._digamma_lambda_item = digamma(self._lambda_item)
        self._mu_user = np.random.gamma(
            self._model.zeta_alpha_shape_prior,
            1.0 / self._model.zeta_alpha_rate_prior,
            size=(self._model.nusers, self._model.latent_dimensions),
        )
        self._log_mu_user = np.log(self._mu_user)
        self._mu_item = np.random.gamma(
            self._model.zeta_beta_shape_prior,
            1.0 / self._model.zeta_beta_rate_prior,
            size=(self._model.nitems, self._model.latent_dimensions),
        )
        self._log_mu_item = np.log(self._mu_item)
        self._xi_user = self._model.zeta_alpha_rate_prior * np.ones(self._model.nusers)
        self._xi_item = self._model.zeta_beta_rate_prior * np.ones(self._model.nitems)
        self._nu_user = (
            self._model.zeta_alpha_shape_prior
            + self._model.latent_dimensions * self._model.alpha_shape_prior
        )
        self._nu_item = (
            self._model.zeta_beta_shape_prior
            + self._model.latent_dimensions * self._model.beta_shape_prior
        )
        self._berpo = self._model.berpo
        self._sum_over_items = np.zeros(
            (self._model.nusers, self._model.latent_dimensions)
        )
        self._sum_over_users = np.zeros(
            (self._model.nitems, self._model.latent_dimensions)
        )
        self._edgelist = data.get_edge_list()
        self._poisson_rate = []
        self._convergence_criterion = convergence_threshold
        self._max_its = max_iterations

    def _update_multinomial_parameters(self):
        """Update the multinomial variational parameters theta and xi."""
        self._sum_over_items = np.zeros(
            (self._model.nusers, self._model.latent_dimensions)
        )
        self._sum_over_users = np.zeros(
            (self._model.nitems, self._model.latent_dimensions)
        )
        self._poisson_rate = []
        for (user, item), count in self._edgelist.items():
            chi = (
                self._digamma_lambda_user[user]
                - self._log_mu_user[user]
                + self._digamma_lambda_item[item]
                - self._log_mu_item[item]
            )
            chi = np.exp(chi)
            chi_normalising = sum(chi)
            self._poisson_rate.append((count, chi_normalising))
            if self._berpo:
                chi /= -np.expm1(-chi_normalising)
            else:
                chi /= chi_normalising
                chi *= float(count)
            self._sum_over_users[item] += chi
            self._sum_over_items[user] += chi

    def _update_user_parameters(self):
        """Update the user variational parameters."""
        self._lambda_user = self._model.alpha_shape_prior + self._sum_over_items
        self._digamma_lambda_user = digamma(self._lambda_user)
        self._mu_user = np.add.outer(
            self._nu_user / self._xi_user,
            (self._lambda_item / self._mu_item).sum(axis=0),
        )
        self._log_mu_user = np.log(self._mu_user)
        self._xi_user = self._model.zeta_alpha_rate_prior + (
            self._lambda_user / self._mu_user
        ).sum(axis=1)

    def _update_item_parameters(self):
        """Update the item variational parameters."""
        self._lambda_item = self._model.beta_shape_prior + self._sum_over_users
        self._digamma_lambda_item = digamma(self._lambda_item)
        self._mu_item = np.add.outer(
            self._nu_item / self._xi_item,
            (self._lambda_user / self._mu_user).sum(axis=0),
        )
        self._log_mu_item = np.log(self._mu_item)
        self._xi_item = self._model.zeta_beta_rate_prior + (
            self._lambda_item / self._mu_item
        ).sum(axis=1)

    def _elbo_theta_terms(self):
        """Terms related to the likelihood for the elbo."""
        term1 = (self._lambda_user / self._mu_user).sum(axis=0)
        term2 = (self._lambda_item / self._mu_item).sum(axis=0)
        elbo_term = -sum(term1 * term2)
        # print(elbo_term)
        for (count, theta) in self._poisson_rate:
            if self._berpo:
                try:
                    with np.errstate(over="raise"):
                        elbo_term += np.log(np.expm1(theta))
                except FloatingPointError:
                    # For large values of theta, then term approximately equivalent to theta.
                    elbo_term += theta
            else:
                elbo_term += count * np.log(theta)
        return elbo_term

    def _elbo_hp_item_terms(self, log_xi, nu_d_xi):
        """Terms related to the item hyper parameters for the elbo."""
        elbo_term = -np.sum(self._model.zeta_beta_shape_prior * log_xi)
        elbo_term -= np.sum(self._model.zeta_beta_rate_prior * nu_d_xi)
        return elbo_term

    def _elbo_item_terms(self):
        """Terms related to the item parameters for the elbo."""

        # beta terms
        log_mu = self._log_mu_item
        log_xi = np.log(self._xi_item)
        nu_d_xi = self._nu_item / self._xi_item
        elbo_term = np.sum(gammaln(self._lambda_item))
        elbo_term -= np.sum(self._lambda_item * log_mu)
        elbo_term += np.sum(
            (self._model.beta_shape_prior - self._lambda_item)
            * (self._digamma_lambda_item - log_mu)
        )
        term = np.transpose(nu_d_xi * np.transpose((1.0 / self._mu_item)))
        elbo_term += np.sum(self._lambda_item * (1.0 - term))
        elbo_term -= (
            self._model.latent_dimensions
            * self._model.beta_shape_prior
            * np.sum(log_xi)
        )
        # xi terms
        elbo_term += self._elbo_hp_item_terms(log_xi, nu_d_xi)
        return elbo_term

    def _elbo_hp_user_terms(self, log_xi, nu_d_xi):
        """Terms related to the user hyper parameters for the elbo."""
        elbo_term = -np.sum(self._model.zeta_alpha_shape_prior * log_xi)
        elbo_term -= np.sum(self._model.zeta_alpha_rate_prior * nu_d_xi)
        return elbo_term

    def _elbo_user_terms(self):
        """Terms related to the user parameters for the elbo."""

        # alpha terms
        log_mu = self._log_mu_user
        log_xi = np.log(self._xi_user)
        nu_d_xi = self._nu_user / self._xi_user
        elbo_term = np.sum(gammaln(self._lambda_user))
        elbo_term -= np.sum(self._lambda_user * log_mu)
        elbo_term += np.sum(
            (self._model.alpha_shape_prior - self._lambda_user)
            * (self._digamma_lambda_user - log_mu)
        )
        term = np.transpose(nu_d_xi * np.transpose((1.0 / self._mu_user)))
        elbo_term += np.sum(self._lambda_user * (1.0 - term))
        elbo_term -= (
            self._model.latent_dimensions
            * self._model.alpha_shape_prior
            * np.sum(log_xi)
        )
        # xi terms
        elbo_term += self._elbo_hp_user_terms(log_xi, nu_d_xi)
        return elbo_term

    def _elbo(self):
        """Evidence lower bound to assess convergence of the variational inference algorithm."""
        elbo = self._elbo_theta_terms()
        elbo += self._elbo_user_terms()
        elbo += self._elbo_item_terms()
        return elbo

    def _change_in_elbo(self, old_elbo, new_elbo):
        change = (new_elbo - old_elbo) / np.abs(old_elbo)
        if change < 0:
            print("ELBO is decreasing", file=sys.stderr)
        return change

    def run_algorithm(self):
        """Run variational inference algorithm until convergence."""
        niterations = 0
        old_elbo = None
        while niterations < self._max_its:
            print(f"Iteration Number {niterations}", file=sys.stderr, end="\r")
            self._update_multinomial_parameters()
            elbo = self._elbo()
            if niterations > 0:
                eps = self._change_in_elbo(old_elbo, elbo)
                # print(f"{eps} {elbo} {old_elbo}", file=sys.stderr)
                if eps <= self._convergence_criterion:
                    break
            old_elbo = elbo
            self._update_user_parameters()
            self._update_item_parameters()
            niterations += 1

        if niterations == self._max_its:
            print(
                f"Algorithm did not reach convergence in {self._max_its} iterations.",
                file=sys.stderr,
            )
        else:
            print(
                f"Algorithm reached convergence in {niterations} iterations. ELBO at convergence {elbo}",
                file=sys.stderr,
            )
        self._model.alpha = self._lambda_user / self._mu_user
        self._model.beta = self._lambda_item / self._mu_item
