"""Unit test for cosmic
"""

__author__ = 'Katie Breivik <katie.breivik@gmail.com>'

import os
import unittest2
import numpy as np
import scipy.integrate
import pandas as pd

import cosmic.Match as Match

np.random.seed(2)
sample = np.random.uniform(0,1,500)

MATCH_TEST = 0.9893134305198774

class TestMatch(unittest2.TestCase):
    """`TestCase` for the match method
    """
    def test_Match(self):
        # test the match by dividing the sample into to sub samples
        # one containing half of the set and one containing the full set

        dataCm = [sample[:int(len(sample)/2)], sample]
        match, bin_width = Match.match(dataCm, 2)
        self.assertAlmostEqual(match[0],MATCH_TEST)

        