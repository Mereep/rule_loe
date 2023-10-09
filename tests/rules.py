"""
# Copyright 2018 Professorship Media Informatics, University of Applied Sciences Mittweida
# licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# @author Richard Vogel, 
# @email: richard.vogel@hs-mittweida.de
# @created: 03.04.2020
"""
import unittest
import numpy as np
from gon import GON
from rule_gon import RuleGon
from hdtree import HDTreeClassifier, TwentyQuantileSplit, EntropyMeasure
from sklearn.metrics import accuracy_score


class RulesTest(unittest.TestCase):

    def setUp(self):
        np.random.seed(0)
        self.X_dummy = np.reshape(np.array([
            np.random.normal(size=(100, 3), scale=0.5) + (0, 0, 0),
            np.random.normal(size=(100, 3), scale=0.5) + (0, 0, 10),
            np.random.normal(size=(100, 3), scale=0.5) + (0, 10, 0),
            np.random.normal(size=(100, 3), scale=0.5) + (0, 10, 10),
            np.random.normal(size=(100, 3), scale=0.5) + (10, 0, 0),
            np.random.normal(size=(100, 3), scale=0.5) + (10, 0, 10),
            np.random.normal(size=(100, 3), scale=0.5) + (10, 10, 0),
            np.random.normal(size=(100, 3), scale=0.5) + (10, 10, 10),
        ]), newshape=(-1, 3))

        self.y_dummy = np.reshape(np.array([np.ones(100) * i for i in range(8)]), -1).astype(str)


    def test_rule_generation(self):
        X = self.X_dummy
        y = self.y_dummy
        cols = ['X', 'Y', 'Z']
        params = dict(allowed_splits=[
            TwentyQuantileSplit
        ],
            information_measure=EntropyMeasure(),
            max_levels=1,
            min_samples_at_leaf=None,
            verbose=False,
            attribute_names=cols
        )

        trees = [HDTreeClassifier(**params) for i in range(4)]

        gon = GON(pool_classifiers=trees,
                  step_size=1,
                  max_jobs=1,
                  val_perc=0.,
                  random_state=42,
                  iterations=8)

        gon.fit(X, y)

        self.assertAlmostEqual(gon.score(X, y), 1, 1, "This problem should be solveable to (almost) 100%")
        rule_pred = RuleGon.from_gon_instance(gon=gon,
                                              max_length_assignment=3)
        y_hat = rule_pred.predict(X)
        self.assertAlmostEqual(gon.score(X, y),
                               accuracy_score(y_true=y, y_pred=y_hat), 1, "Approximated rules should almost have the"
                                                                       "same performance as GoN itself")

        # remove rule with minimal coverage
        n_rules_before = len(rule_pred.concepts_and_rules)
        min_cov = rule_pred.get_min_coverage()
        with self.assertRaises(ValueError):
            # should try to remove all rules, since they all have the same coverage -> will raise
            rule_pred.remove_under_coverage(min_cov+1e-5)

        target_to_remove = rule_pred.concepts_and_rules[0][0].target_concept
        rule_pred.remove_rule(0)
        n_rules_after = len(rule_pred.concepts_and_rules)
        self.assertEqual(n_rules_before - 1, n_rules_after, "There should be one rule removed from the system")

        y_hat = rule_pred.predict(X)
        self.assertNotIn(target_to_remove, y_hat, f"Target {target_to_remove} should have been removed, hence"
                                                  f"cannot be predicted anymore (there should be only one rule"
                                                  f"predicting it before)")

        self.assertGreater(gon.score(X, y),
                           accuracy_score(y_true=y, y_pred=y_hat),
                           "After removing rules the score should decrease")