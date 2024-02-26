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
# @created: 20.05.2020
"""
import dataclasses
import typing

from loe import LoE
from hdtree import HDTreeClassifier, Node, simplify_rules, FixedValueSplit, TwentyQuantileSplit, TenQuantileRangeSplit, \
    EntropyMeasure, GiniMeasure, AbstractSplitRule
from typing import List, Optional
import numpy as np
import dacite

# DATA STRUCTURES & CONSTANTS
LABEL_EXPERT = 'Expert'
LABEL_NO_EXPERT = 'No Expert'

@dataclasses.dataclass(frozen=True)
class Concept:
    """ A Concept is a dpecific rule with meta data and label """

    """ label of the concept """
    target_concept: str

    """ all nodes of the concept (expert + assignment tree) """
    nodes_complete: List[Node]

    """ only expert / nerd nodes """
    nodes_expert: List[Node]

    """ only assignment nodes """
    nodes_assignment: List[Node]

    """ merged nodes (be aware those may stem from the nerd tree
    OR the assignment tree). Those were used to generate the readable_rules """
    nodes_simplified: List[Node]

    """ rules of nodes as string evaluated for a specific sample """
    readable_rules: List[str]

    """ coverage of the rules (# of samples where rules apply) """
    coverage: int

    """ amount of rules that is covered and meet target_concept """
    precision: float

    """ if concepts derives from a simplified (pruned) tree """
    simplified_tree: bool

    """ original index of the nerd of the LoE where the info derived from """
    nerd_idx: int

    """ which of the experts' features are used in assignment tree? """
    assignment_feature_mask: list[bool]

    """ a sample which meets all the conditions (was used to evaluate)"""
    sample_dummy: List[typing.Any]

    """ indicates if the node came from the original nerd tree.
    Be aware that non-nerd trees may not see all features (feature mask has to be used) 
    """
    simplified_node_came_from_nerd_tree: List[bool]

    """ original attributes / features when deriving the concept"""
    original_attribute_names: List[str]

def generate_assignment_trees_iterator(loe: LoE,
                                       feature_names: List[str],
                                       only_use_important_attributes: bool = True,
                                       only_for_model_indices: Optional[List[int]] = None,
                                       allowed_splits: Optional[AbstractSplitRule] = None,
                                       max_levels: int = 4) -> List[HDTreeClassifier]:
    """
    Will generate a decision tree that learns the assignment function from data point to model

    :param loe: LoE model
    :param feature_names: names for each feature has to match amount of features
    :param only_use_important_attributes:
    :param only_for_model_indices: will not create an assignment tree for all LoE models (default) but only for listed
    :param allowed_splits:
    :param max_levels: how much to grow max?
    :return:
    """

    allowed_splits = allowed_splits or [FixedValueSplit.build(),
                                        TwentyQuantileSplit.build(),
                                        TenQuantileRangeSplit.build()
    ]

    all_data = loe.get_expert_data()
    assignments = loe.assign_data_points_to_model(loe.get_train_data(processed=True),
                                                  is_processed=True)

    if only_use_important_attributes is True:
        importance = loe.get_important_feature_indexer()
    else:
        importance = np.ones(len(feature_names),
                             dtype=bool)

    params_for_tree = dict(max_levels=max_levels,
                           allowed_splits=allowed_splits,
                           information_measure=GiniMeasure(),
                           attribute_names=np.array(feature_names)[importance],
                           min_samples_at_leaf=min(5, len(loe.get_DSEL())),
                           verbose=False)

    trees = []
    for model_idx, assignment in assignments.items():
        if not only_for_model_indices or model_idx in only_for_model_indices:
            indexer_for_model = np.zeros(len(all_data),
                                         dtype=bool)
            indexer_for_model[assignment] = True
            tree = HDTreeClassifier(**params_for_tree)
            tree.fit(all_data[:, importance], np.where(indexer_for_model,
                                                       LABEL_EXPERT,
                                                       LABEL_NO_EXPERT))
            trees.append(tree)

            yield tree


def generate_assignment_trees(loe: LoE,
                              feature_names: List[str],
                              only_use_important_attributes: bool = True,
                              only_for_model_indices: Optional[List[int]] = None,
                              max_levels: int = 4,
                              allowed_splits: Optional[AbstractSplitRule] = None,
                              ) -> List[HDTreeClassifier]:
    """
    A wrapper of @see generate_assignment_trees_generate_assignment_trees_iterator

    :param loe:
    :param feature_names:
    :param only_use_important_attributes:
    :param only_for_model_indices:
    :param max_levels:
    :param allowed_splits
    :return:
    """
    return [*generate_assignment_trees_iterator(feature_names=feature_names,
                                                loe=loe,
                                                only_use_important_attributes=only_use_important_attributes,
                                                only_for_model_indices=only_for_model_indices,
                                                allowed_splits=allowed_splits,
                                                max_levels=max_levels)]

def gather_concepts(loe: LoE,
                    feature_names: List[str],
                    use_simplified_trees: bool = False,
                    nerd_trees: List[HDTreeClassifier] = None,
                    max_levels: int = 4,
                    allowed_splits: Optional[List[AbstractSplitRule]] = None,
                    ass_trees: List[HDTreeClassifier] = None) -> List[Concept]:
    """
    :param loe:
    :param feature_names: names for each feature (complete features, do not care for importance-handling)
    :param use_simplified_trees: Will prune the trees before extracting rules. If nerd_trees and assignment trees are
    this parameter is ignored
    :param ass_trees: trees that model the decision process for assignment / selection of LoE
    :param allowed_splits
    :param max_levels:assignment tree max depth

    :return:
    """

    if nerd_trees is not None:
        trees = nerd_trees
    else:
        trees = loe.pool_classifiers_
        if use_simplified_trees:
            trees = [tree.simplify(return_copy=True) for tree in trees]

    if ass_trees is not None:
        assert len(ass_trees) == len(trees), "Amount of assignment trees and nerd" \
                                             " trees does not match (Code: 893478923)"
    else:
        ass_trees = generate_assignment_trees(loe=loe, feature_names=feature_names,
                                              only_use_important_attributes=True,
                                              allowed_splits=allowed_splits,
                                              max_levels=max_levels)
        if use_simplified_trees:
            ass_trees = [tree.simplify(return_copy=False) for tree in ass_trees]

    expert_data_loe = loe.get_expert_data()
    concepts: List[Concept] = []

    for tree_index, ass_tree in enumerate(ass_trees):
        nerd_tree: HDTreeClassifier = trees[tree_index]
        agreeing_sample_indices_assignment_all_data: List[int] = []
        assignment_conditions_nodes: List[List[Node]] = []

        # nerd_conditions_nodes: List[List[Node]] = []

        # assignment rules
        leafs_assignment: List[Node] = [node for node in ass_tree.get_all_nodes_below_node(node=None) if node.is_leaf()]

        if len(leafs_assignment) == 0:
            leafs_assignment = [ass_tree.get_head()]

        for leaf_nerd in leafs_assignment:
            concept = ass_tree.get_prediction_for_node(leaf_nerd)  # the actual class

            if concept == LABEL_EXPERT:
                chain = ass_tree.follow_node_to_root(node=leaf_nerd) + [leaf_nerd]
                agreeing_sample_indices_assignment_all_data += list(leaf_nerd.get_data_indices())
                assignment_conditions_nodes.append(chain)

        # nerd rules
        leafs_nerd = [node for node in nerd_tree.get_all_nodes_below_node(node=None) if node.is_leaf()]

        if len(leafs_nerd) == 0:
            leafs_nerd = [nerd_tree.get_head()]

        # follow each experts' leaf and cross join with each assignment trees' path to expert
        for leaf_nerd in leafs_nerd:
            # get the leafs decision
            concept = nerd_tree.get_prediction_for_node(node=leaf_nerd)
            concept_readable = loe.enc_.inverse_transform([concept])[0]
            nerd_chain = nerd_tree.follow_node_to_root(leaf_nerd) + [leaf_nerd]

            # now check each possible way that lands at that expert
            for assignment_option_idx in range(0, len(assignment_conditions_nodes)):

                # get all samples withing current assignment node
                assignment_leaf_node = assignment_conditions_nodes[assignment_option_idx][-1]
                assignment_sample = loe.get_expert_data()[assignment_leaf_node.get_data_indices()]
                assignment_targets = loe.get_train_targets()[assignment_leaf_node.get_data_indices()]

                # get all samples that are in assignment model AND follow expert path to current leaf
                flow_samples_mask = [nerd_tree.extract_node_chain_for_sample(dp)[-1] == leaf_nerd for dp in
                                     assignment_sample]

                if sum(flow_samples_mask) == 0:
                    continue

                # same_flow = assignment_sample[flow_samples_mask]
                y_sample = assignment_targets[flow_samples_mask].astype(str)
                prec = sum(y_sample == [str(concept)]) / sum(flow_samples_mask)
                cov = sum(flow_samples_mask) / len(expert_data_loe)

                # option_ass_conditions_nodes = assignment_conditions_nodes[assignment_option_idx]
                nodes_complete = assignment_conditions_nodes[assignment_option_idx][:-1] + nerd_chain[:-1]
                rules_complete = [node.get_split_rule() for node in nodes_complete if node.get_split_rule() is not None]

                data_point = expert_data_loe[assignment_leaf_node.get_data_indices()][flow_samples_mask][0]
                rules_simplified = simplify_rules(rules=rules_complete,
                                                  model_to_sample={nerd_tree: data_point,
                                                                   ass_tree: data_point[
                                                                       loe.get_important_feature_indexer()
                                                                   ]}
                                                  )

                chain = nerd_tree.follow_node_to_root(node=leaf_nerd)
                # nerd_conditions_nodes.append(chain)

                readable_rules = [
                    rule.explain_split(sample=data_point if rule.get_tree() is nerd_tree else data_point[
                        loe.get_important_feature_indexer()], hide_sample_specifics=True)
                    for rule in rules_simplified]

                # gather expected rule outcomes assignment trees on sample
                is_nerd_tree = [rule.get_tree() is nerd_tree for rule in rules_simplified]
                nodes_simplified = [rule.get_node() for rule in rules_simplified
                                      if rule.get_node().get_split_rule() is not None]
                concept_description = Concept(
                    target_concept=concept_readable,
                    nodes_complete=nodes_complete,
                    nodes_expert=chain,
                    nodes_assignment=assignment_conditions_nodes[assignment_option_idx],
                    nodes_simplified=nodes_simplified,
                    simplified_node_came_from_nerd_tree=is_nerd_tree,
                    readable_rules=readable_rules,
                    precision=prec,
                    coverage=cov,
                    simplified_tree=use_simplified_trees,
                    nerd_idx=tree_index,
                    assignment_feature_mask=list(loe.get_important_feature_indexer()),
                    sample_dummy=list(data_point),
                    original_attribute_names=feature_names,
                )

                concepts.append(concept_description)

    return concepts