from __future__ import division

from model import *
import config
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
import logging


def evaluate(model, dataset, verbose=True):
    if verbose:
        logging.info('evaluating [%s] dataset, [%d] examples' % (dataset.name, dataset.count))

    exact_match_ratio = 0.0

    for example in dataset.examples:
        logging.info('evaluating example [%d]' % example.eid)
        hyps, hyp_scores = model.decode(example, max_time_step=MAX_EXAMPLE_ACTION_NUM)
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

def evaluate_decode_results(dataset, decode_results):
    assert dataset.count == len(decode_results)
    cum_oracle_bleu = 0.0
    cum_oracle_acc = 0.0
    cum_bleu = 0.0
    cum_acc = 0.0
    sm = SmoothingFunction()

    all_references = []
    all_predictions = []

    for eid in range(dataset.count):
        example = dataset.examples[eid]
        refer_tree = example.parse_tree
        refer_ast_tree = tree_to_ast(refer_tree)
        refer_source = astor.to_source(refer_ast_tree)
        refer_tokens = tokenize(refer_source)

        decode_cands = decode_results[eid]
        decode_cand = decode_cands[3]

        cid, cand, ast_tree, code = decode_cand
        predict_tokens = tokenize(code)

        if refer_tokens == predict_tokens:
            cum_acc += 1

        all_references.append([refer_tokens])
        all_predictions.append(predict_tokens)

        score = sentence_bleu([refer_tokens], predict_tokens, smoothing_function=sm.method3)
        print 'raw_id: %d, score: %f' % (example.raw_id, score)
        # print code
        cum_bleu += score

        # compute oracle
        best_score = -1000
        cur_oracle_acc = 0.
        for decode_cand in decode_cands[:10]:
            cid, cand, ast_tree, code = decode_cand
            try:
                predict_tokens = tokenize(code)
            except:
                continue

            if predict_tokens == refer_tokens:
                cur_oracle_acc = 1

            score = sentence_bleu([refer_tokens], predict_tokens, smoothing_function=sm.method3)
            if score > best_score:
                best_score = score

        cum_oracle_bleu += best_score
        cum_oracle_acc += cur_oracle_acc

    print 'corpus level bleu: %f' % corpus_bleu(all_references, all_predictions, smoothing_function=sm.method3)
    print 'sentence level bleu: %f' % (cum_bleu / dataset.count)
    print 'accuracy: %f' % (cum_acc / dataset.count)
    print 'oracle bleu: %f' % (cum_oracle_bleu / dataset.count)
    print 'oracle accuracy: %f' % (cum_oracle_acc / dataset.count)