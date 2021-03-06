#! /usr/bin/env python3
"""This module contains all methods to store and read in the edgelist for the PMF package."""
import logging

logger = logging.getLogger(__name__)


class Data:
    """Data stores the edge lists for the PMF model."""

    def __init__(self, edgelist=None, edgelist_path=None, userlist=None, itemlist=None):
        """The constructor for Data class. Either edgelist or edgelist_path must be specified.
        If both are provided edgelist_path will take precedence. You can optionally
        pass in a list of user and item identifiers. The integer identifiers will
        be created from these lists if provided, allowing user and items with no
        observed edges.

        Parameters:
        edgelist_path (str): The filepath for the edgelist with schema user, item, count.
        edgelist (list): List with each element as (user, item, optional count). If count
                         not given set at one.
        userlist (list): A list of user identifiers.
        itemlist (list): A list of item identifiers.
        """
        if edgelist is None and edgelist_path is None:
            raise ValueError("Either edgelist or edgelist_path needs to be provided.")
        self.user_hash = {}
        self._user_hash_rev = {}
        self.item_hash = {}
        self._item_hash_rev = {}
        self._edge_list = {}
        userlist_data = None
        itemlist_data = None
        if edgelist_path is not None:
            edgelist, userlist_data, itemlist_data = self._read_edgelist_from_file(
                edgelist_path
            )
        if userlist is None:
            userlist = userlist_data
        if itemlist is None:
            itemlist = itemlist_data

        self._parse_edge_list(edgelist, userlist, itemlist)

    def _read_edgelist_from_file(self, edgelist_fp):
        userlist = set([])
        itemlist = set([])
        with open(edgelist_fp, "r", encoding="utf-8") as fhandle:
            edgelist = []
            for line in fhandle:
                fields = line.rstrip("\r\n").split(",")
                user = fields[0]
                userlist.add(str(user))
                item = fields[1]
                itemlist.add(str(item))
                count = 1
                if len(fields) > 2:
                    count = int(fields[2])
                edgelist.append([user, item, count])
        return edgelist, userlist, itemlist

    def _parse_edge_list(self, edgelist, userlist, itemlist):
        """Parse edgelist and create integer unique identifiers for the users and items."""
        if userlist is None:
            userlist = {edge[0] for edge in edgelist}
        if itemlist is None:
            itemlist = {edge[1] for edge in edgelist}
        self.user_hash = {str(j): i for i, j in enumerate(sorted(userlist))}
        self.item_hash = {str(j): i for i, j in enumerate(sorted(itemlist))}
        for edge in edgelist:
            user = edge[0]
            item = edge[1]
            count = 1
            if len(edge) > 2:
                count = edge[2]
            self._edge_list[
                (self.user_hash[str(user)], self.item_hash[str(item)])
            ] = count
        logger.info("Read in edge list")
        logger.info("Number of users: %s", len(self.user_hash))
        logger.info("Number of items: %s", len(self.item_hash))
        logger.info("Total number of edges: %s", len(self._edge_list))
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
        user = str(user)
        if user not in self.user_hash:
            return None

        return self.user_hash[user]

    def get_id_for_item(self, item):
        """Return item identifier given original identifier."""
        item = str(item)
        if item not in self.item_hash:
            return None

        return self.item_hash[item]
