#! /usr/bin/env python3
"""This module contains all methods to store and read in the edgelist for the PMF package."""
import sys


class Data:
    """Data stores the edge lists for the PMF model."""

    def __init__(self, edgelist=None, edgelist_path=None):
        """The constructor for Data class. Either edgelist or edgelist_path must be specified.
        If both are provided edgelist_path will take precedence.

        Parameters:
        edgelist_path (str): The filepath for the edgelist with schema user, item, count.
        edgelist (list): List with each element as (user, item, count).
        """
        if edgelist is None and edgelist_path is None:
            print(
                "Either filepath to edgelist or edgelist itself must be provided.",
                file=sys.stderr,
            )
        self.user_hash = {}
        self._user_hash_rev = {}
        self.item_hash = {}
        self._item_hash_rev = {}
        self._edge_list = {}
        if edgelist_path is not None:
            edgelist = self._read_edgelist_from_file(edgelist_path)

        self._parse_edge_list(edgelist)

    def _read_edgelist_from_file(self, edgelist_fp):
        with open(edgelist_fp, "r", encoding="utf-8") as edgelist:
            edgelist = []
            for line in edgelist:
                fields = line.rstrip("\r\n").split(",")
                user = fields[0]
                item = fields[1]
                count = fields[2]
                edgelist.append([user, item, count])
        return edgelist

    def _parse_edge_list(self, edgelist):
        """Parse edgelist and create integer unique identifiers for the users and items."""
        user_counter = 0
        item_counter = 0
        for user, item, count in edgelist:
            if user not in self.user_hash:
                self.user_hash[user] = user_counter
                user_counter += 1

            if item not in self.item_hash:
                self.item_hash[item] = item_counter
                item_counter += 1

            self._edge_list[(self.user_hash[user], self.item_hash[item])] = count
        print(
            f"Read in edge list.\nNumber of users: {len(self.user_hash)} \
                    \nNumber of items: {len(self.item_hash)}",
            file=sys.stderr,
        )
        self._user_hash_rev = {v: k for k, v in self.user_hash.items()}
        self._item_hash_rev = {v: k for k, v in self.item_hash.items()}

    def get_user_id_mappings(self):
        """Return dict mapping user integer identifier to original identifier."""
        return self._user_hash_rev

    def get_item_id_mappings(self):
        """Return dict mapping item integer identifiers to original identifier."""
        return self._item_hash_rev

    def get_edge_list(self):
        """Return the edge list."""
        return self._edge_list

    def get_number_users(self):
        """Return the total number of unique users."""
        return len(self.user_hash)

    def get_number_items(self):
        """Return the total number of unique items."""
        return len(self.item_hash)

    def get_user_from_id(self, uid):
        """Return user original identifier given user integer identifier."""
        if uid not in self._user_hash_rev:
            return None

        return self._user_hash_rev[uid]

    def get_item_from_id(self, iid):
        """Return item original identifier given item integer identifier."""
        if iid not in self._item_hash_rev:
            return None

        return self._item_hash_rev[iid]

    def get_id_for_user(self, user):
        """Return user integer identifier given original identifier."""
        if user not in self.user_hash:
            return None

        return self.user_hash[user]

    def get_id_for_item(self, item):
        """Return item identifier given original identifier."""
        if item not in self.item_hash:
            return None

        return self.item_hash[item]
