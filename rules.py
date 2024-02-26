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
# @author Richard Vogel
# @email: richard.vogel@hs-mittweida.de
# @created: 03.04.2020
"""
from rule_loe import Concept, gather_concepts, simplify_rules
from typing import Optional, List, Tuple, Dict, Callable
import numpy as np
from hdtree import AbstractSplitRule
from loe import LoE
from hdtree import HDTreeClassifier
from functools import reduce
from sklearn.utils import check_X_y
from sklearn.metrics import accuracy_score
import pandas as pd
from collections import OrderedDict


class Condition:
    """
    Represents one single condition
    """
    @classmethod
    def from_split_rule(cls, rule: AbstractSplitRule,
                        *args, **kwargs) -> 'Condition':
        """
        Will build a rule that is represents a split within a tree
        :param rule:
        :param filter_columns: if given, before the actual rule is applied, that boolean filter will be used
        :return:
        """
        rule_logic = lambda sample: rule.get_child_node_index_for_sample(sample=sample)
        me = cls(rule_logic=rule_logic, *args, **kwargs)

        return me

    def __init__(self,  rule_logic: Callable[[np.ndarray], int],
                 filter_columns: Optional[np.ndarray] = None):
        self._filter_columns = filter_columns
        if self.needs_filter():
            self._rule_logic = lambda X: rule_logic(X[self._filter_columns])
        else:
            self._rule_logic = rule_logic

    def needs_filter(self) -> bool:
        return self._filter_columns is not None

    def __call__(self, X: np.ndarray) -> int:
        """
        Will apply the rule and returns the option index this sample applies to
        :param X:
        :return:
        """
        return self._rule_logic(X)


class RuleClause:
    """
    Represents a list of AND-connected Rules
    """
    def __init__(self, rules: List[Condition], expected_rule_outcomes: List[int]):
        self._rules = rules
        self._expected_outcomes = expected_rule_outcomes

    def apply_for_sample(self, dp) -> List[int]:
        return [rule(dp) for rule in self._rules]

    def __len__(self):
        return len(self._rules)

    def applies_to_sample(self, X: np.ndarray) -> bool:
        """
        Checks if each rules' expected outcome matches the sample outcome
        :return:
        """
        return np.array_equal(self.apply_for_sample(X), self._expected_outcomes)

    def get_expected_outcome(self) -> List[int]:
        """
        A sample has to follow the same expected outcome to hit the rule
        :return:
        """
        return self._expected_outcomes

    def relative_hit(self, dp: np.ndarray) -> float:
        """
        Will return the proportion of rules hit
        
        :param dp:
        :return:
        """
        equals = np.equal(self.get_expected_outcome(),
                          self.apply_for_sample(dp=dp))

        return sum(equals) / len(self.get_expected_outcome())


class RuleLoE:
    def __init__(self, concepts: List[Concept], min_precision: float = 0.5,
                 min_coverage: float = 0.01):
        """
        Will fit the Rule Predictor on the given concepts and rules
        before storing those they are filtered as requested and ordered by accuracy


        :param concepts:
        :param min_precision:
        :param min_coverage:
        """
        assert 0 <= min_precision <= 1, "The minimal precision has to be in range [0, ..., 1] (Code: 3204823)"

        concepts_and_rules = [*zip(concepts, rules_from_concepts(concepts))]

        # sort by precision of rule
        def sort_precision_coverage(concept_rule):
            concept, rule_chain = concept_rule
            return concept.precision, concept.coverage

        self.concepts_and_rules: List[Tuple[Concept, RuleClause]] = sorted([c_r for c_r in concepts_and_rules if c_r[0].precision >= min_precision and c_r[0].coverage >= min_coverage],
                                         key=sort_precision_coverage,
                                         reverse=True)
        if len(self.concepts_and_rules) == 0:
            raise ValueError(f"The rule filter with min precision {min_precision} and min coverage {min_coverage} "
                             f"leaves no rules left over (Code: 234233446354)")

    def predict(self, X_expert: np.ndarray, positive_class: Optional[str]=None, negative_class: Optional[str]=None):
        """
        Will generate a prediction by following these steps
        1) Order rules by precision
        2) From Top to bottom find first matching rule
        3) if none found: Repeat and find best relative rule overlap (most attributes match)

        if positive class is given the predictor will only use the rules that have the outcome
        of the positive class. otherwise it will return the other outcome.
        (May have to be supplied explicitly if rules do not contain that other outcome anymore)

        :param X_expert:
        :param positive_class: Only supply if you have exactly two outcomes and want to ignore all other rules
        :return:
        """
        X_expert = np.atleast_2d(X_expert)
        res = np.ndarray((len(X_expert),), dtype=np.object_)
        res[:] = ''
        possible_outcomes = None
        if positive_class is not None:
            possible_outcomes = [*self.get_unique_labels().keys()]
            assert len(possible_outcomes) <= 2, "The rule list has to consist of exactly " \
                                                "two or 1 rules if in binary mode (Code: 32482390)"
            if len(possible_outcomes) == 1:
                assert negative_class is not None, "The predictor is in binary mode. However, since your resulting" \
                                                   "rules only contain one outcome, you have to supply the name of" \
                                                   "the other class explicitly (Code: 3209482390)"

        for i in range(len(X_expert)):
            pred = self.get_best_fitting_rule(dp=X_expert[i], positive_class=positive_class)
            if pred is None:
                # Should only happen in two class mode (should not be thrown to user unless there is a bug)
                assert positive_class is not None, "No rule found, this should only happen in two class " \
                                                   "mode. This is an implementation error (Code: 324082390)"

                if len(possible_outcomes) == 2:
                    # just return the other possible outcome
                    if possible_outcomes[0] == positive_class:
                        res[i] = possible_outcomes[1]
                    else:
                        res[i] = possible_outcomes[0]
                elif len(possible_outcomes) == 1:
                    res[i] = negative_class
                else: # cannot happen, since #outcomes is checked before
                    raise Exception("Report me. You should not see this (Code: 29034820394)")

            else:
                concept, clause = pred
                res[i] = concept.target_concept

        return res.astype(str)

    def get_unique_labels(self) -> Dict[str, float]:
        """
        Will return all possible outcomes
        :return:
        """
        d = {}

        for rule_idx, concept_rule in enumerate(self.concepts_and_rules):
            concept, rule = concept_rule
            if concept.target_concept not in d:
                d[concept.target_concept] = 0
            d[concept.target_concept] += concept.coverage

        return d

    @classmethod
    def from_loe_instance(cls, loe: LoE, max_length_assignment: int = 4, *args, **kwargs):
        assert loe.is_fitted(), "LoE has to be fit on data (Code: 234982340)"
        assert np.all([isinstance(clf, HDTreeClassifier) for clf in loe.pool_classifiers_]), "Only HDTrees are supported" \
                                                                                             " atm (Code: 98490238)"

        concepts = gather_concepts(loe, feature_names=loe.pool_classifiers_[0].get_attribute_names(),
                                   use_simplified_trees=True,
                                   max_levels=max_length_assignment)

        me = cls(concepts=concepts, *args, **kwargs)

        return me

    def get_best_fitting_rule(self, dp: np.ndarray,
                              positive_class: Optional[str] = None) -> Optional[Tuple[Concept, RuleClause]]:
        """
        Gets the rule with the best complete cover or best relative overlap. On two class-mode if there is no
        complete cover for the positive class, None is returned

        :param dp:
        :param positive_class:
        :return:
        """
        assert len(dp.shape) == 1, "please provide exactly one data point to this method (Code: 902348230)"

        best_rule_idx = -1
        best_dist = float('-inf')
        best_coverage = 0       # not needed atm, just kept here for easier adoption to other logic
        best_precision = 0

        # note that they are ordered
        for rule_idx, concept_rule in enumerate(self.concepts_and_rules):
            concept, rule = concept_rule
            curr_class = concept.target_concept
            curr_coverage = concept.coverage
            curr_precision = concept.precision

            if positive_class is None or positive_class == curr_class:
                if rule.applies_to_sample(dp):
                    best_rule_idx = rule_idx
                    # we found the first hitting rule -> since they're ordered by precision we take that as granted
                    break
                else:
                    # rule does not hit -> we to what degree it hits still
                    # so if we do not find any perfect rule we take the one with the best score
                    if positive_class is None:  # relative rule hitting not done in binary mode
                        hit_prop = rule.relative_hit(dp)
                        # we will update if a better relative hit is done
                        # on tie we will use the one with the best coverage
                        if hit_prop > best_dist or (hit_prop == best_dist and curr_precision > best_precision):
                            best_rule_idx = rule_idx
                            best_dist = hit_prop
                            best_coverage = curr_coverage

        if best_rule_idx == -1:  # no rule found
            if positive_class is not None:
                return None
            else:
                best_rule_idx = 0

        return tuple(self.concepts_and_rules[best_rule_idx])

    def remove_under_coverage(self, min_coverage:float):
        """
        Will remove all rules that do not hit at least min_coverage proportions of the data

        :param min_coverage:
        :return:
        :raises: ValueError if there would be no rules left after removing
        """
        assert 0 < min_coverage < 1, "min coverage has to be in ]0,...,1[ (Code: 923482093)"
        filtered = [*filter(lambda concept_rule: concept_rule[0].coverage >= min_coverage,
                            self.concepts_and_rules)]

        if len(filtered) == 0:
            raise ValueError("min coverage is too high, "
                             "after applying there would be no rules left (Code: 3982390)")

        self.concepts_and_rules = filtered

    def get_min_coverage(self) -> float:
        """
        Will find the value of the rule with minimum coverage
        :return:
        """
        return reduce(lambda before, concept_rule: concept_rule[0].coverage if concept_rule[0].coverage > before else before,
                      self.concepts_and_rules,
                      0)

    def count_conditions(self) -> int:
        """
        Will get the total amount of conditions within all rules
        :return:
        """
        i = 0
        for (concept, rules) in self.concepts_and_rules:
            i += len(concept.readable_rules)

        return i

    def count_rules(self) -> int:
        """
        Will return amount of rules
        :return:
        """
        return len(self.concepts_and_rules)

    def calc_avg_rule_length(self) -> float:
        """
        returns how long each rule is in average
        :return:
        """
        return self.count_conditions() / self.count_rules()

    def remove_rule(self, rule_idx: int):
        """
        Will remove a specific rule from the system
        :param rule_idx:
        :return:
        """
        assert 0 <= rule_idx < len(self.concepts_and_rules) and len(self.concepts_and_rules) > 1, \
            "The index has to be available and you cannot remove the last rule (Code: 2380293)"

        del self.concepts_and_rules[rule_idx]

    def score(self, X_expert: np.ndarray, y: np.ndarray, positive_class: Optional[str]=None,
              negative_class: Optional[str]=None):

        X_expert, y = check_X_y(X_expert, y)
        pred = self.predict(X_expert=X_expert, positive_class=positive_class, negative_class=negative_class)
        return accuracy_score(y_true=y, y_pred=pred)

    def explain(self, only_for_concept: Optional[str] = None) -> pd.DataFrame:
        if only_for_concept is not None:
            relevant = [*filter(lambda c_r: str(c_r[0].target_concept) == only_for_concept, self.concepts_and_rules)]
            if len(relevant) == 0:
                raise Exception(f"There are no rules that have the concept {only_for_concept} (Code: 828349023)")
        else:
            relevant = self.concepts_and_rules

        # sort by coverage
        relevant = sorted(relevant, key=lambda c_r: c_r[0].coverage, reverse=True)

        # put into bins
        avail_concepts = set([c_r[0].target_concept for c_r in relevant])
        data = []
        max_rules_length = reduce(lambda prev, c_r: max(prev, len(c_r[0].readable_rules)), relevant, 0)

        for target in avail_concepts:
            filtered_by_concept = [c_r for c_r in relevant if c_r[0].target_concept == target]
            for c_r in filtered_by_concept:
                concept: Concept = c_r[0]
                rules_dict = {f'Rule {i + 1}': concept.readable_rules[i] if i < len(concept.readable_rules) else ''
                              for i in range(max_rules_length)}

                info_dict = {
                    'Coverage in Percent': round(concept.coverage * 100, 2),
                    'Precision in Percent': round(concept.precision * 100, 2),
                    'Nerd': '#'+ str(concept.nerd_idx+1),
                    'Prediction': str(concept.target_concept)}

                data.append(OrderedDict(**rules_dict,
                            **info_dict))


        return pd.DataFrame(data)

    def calculate_rule_coverage(self, X_expert: np.ndarray) -> float:
        """
        Checks what proportion of the sample is covered by a rule

        :param X_expert:
        :return:
        """
        X_expert = np.atleast_2d(X_expert)
        assert len(X_expert) > 0, "You have to provide at least one sample"

        covered = 0
        i = 0
        for i in range(len(X_expert)):
            has_a_rule = False
            for c_r in self.concepts_and_rules:
                concept, rule = c_r
                if rule.applies_to_sample(X_expert[i]):
                    has_a_rule = True
                    break

            if has_a_rule:
                covered += 1.

        return covered / (i+1)          # cannot divide by zero

    def calculate_agreement_with_loe(self, loe: LoE, X: np.ndarray, X_expert: Optional[np.ndarray]=None) -> float:
        """
        Will calculate the proportion of the data where LoE and ourselves have the
        same outcome (prediction)
        :param loe:
        :param X:
        :param X_expert:
        :return:
        """
        X = np.atleast_2d(X)
        if X_expert is None:
            X_expert = X

        X_expert = np.atleast_2d(X_expert)

        assert len(X) == len(X_expert), "X and X expert differ in length (Code: 3204823094)"

        y_loe = loe.predict(X=X, X_expert=X_expert).astype(str)
        y_self = self.predict(X_expert)

        return sum(y_loe == y_self) / len(y_loe)


def rules_from_concepts(concepts: List[Concept]) -> List[RuleClause]:
    """
    Transforms Concepts into Rules
    :param concepts:
    :return:
    """
    rules = []
    for concept in concepts:
        nerd_split_rules = [node.get_split_rule() for node in concept.nodes_expert if not node.is_leaf()]
        ass_split_rules = [node.get_split_rule() for node in concept.nodes_assignment if not node.is_leaf()]
        #nerd_split_rules = simplify_rules(nerd_split_rules)
        #ass_split_rules = simplify_rules(ass_split_rules)

        ass_feature_mask = concept.assignment_feature_mask

        rules_nerd = [Condition.from_split_rule(rule=sr) for sr in nerd_split_rules]
        rules_ass = [Condition.from_split_rule(rule=sr, filter_columns=ass_feature_mask) for sr in ass_split_rules]

        rules_complete = rules_nerd + rules_ass
        expected_outcome = [rule(np.array(concept.sample_dummy)) for rule in rules_complete]

        chain = RuleClause(rules=rules_complete,
                           expected_rule_outcomes=expected_outcome)

        rules.append(chain)

    return rules
