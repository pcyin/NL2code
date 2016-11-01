from __future__ import division

from model import *
import config
import logging


def evaluate(model, dataset, verbose=True):
    if verbose:
        logging.info('evaluating [%s] dataset, [%d] examples' % (dataset.name, dataset.count))

    exact_match_ratio = 0.0

    for example in dataset.examples:
        logging.info('evaluating example [%d]' % example.eid)
        hyps, hyp_scores = model.decode(example, max_time_step=MAX_EXAMPLE_RULE_NUM)
        gold_rules = example.rules

        if len(hyps) == 0:
            logging.warning('no decoding result for example [%d]!' % example.eid)
            continue

        best_hyp = hyps[0]
        predict_rules = [dataset.grammar.id_to_rule[rid] for rid in best_hyp]

        assert len(predict_rules) > 0 and len(gold_rules) > 0

        exact_match = sorted(gold_rules, key=lambda x: x.__repr__()) == sorted(predict_rules, key=lambda x: x.__repr__())
        if exact_match:
            exact_match_ratio += 1

        # p = len(predict_rules.intersection(gold_rules)) / len(predict_rules)
        # r = len(predict_rules.intersection(gold_rules)) / len(gold_rules)

    exact_match_ratio /= dataset.count

    logging.info('exact_match_ratio = %f' % exact_match_ratio)

    return exact_match_ratio
