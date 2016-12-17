# -*- coding: UTF-8 -*-

from __future__ import division

from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
import logging
import traceback

from nn.utils.generic_utils import init_logging

from model import *


def tokenize_for_bleu_eval(code):
    code = re.sub(r'([^A-Za-z0-9_])', r' \1 ', code)
    code = re.sub(r'([a-z])([A-Z])', r'\1 \2', code)
    code = re.sub(r'\s+', ' ', code)
    code = code.replace('"', '`')
    code = code.replace('\'', '`')
    tokens = [t for t in code.split(' ') if t]

    return tokens

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

def evaluate_decode_results(dataset, decode_results, verbose=True):
    from lang.py.parse import tokenize_code, de_canonicalize_code
    # tokenize_code = tokenize_for_bleu_eval
    import ast
    assert dataset.count == len(decode_results)

    f = f_decode = None
    if verbose:
        f = open(dataset.name + '.exact_match', 'w')
        exact_match_ids = []
        f_decode = open(dataset.name + '.decode_results.txt', 'w')
        eid_to_annot = dict()

        if MODE == 'django':
            for raw_id, line in enumerate(open('/Users/yinpengcheng/Research/SemanticParsing/CodeGeneration/en-django/all.anno')):
                eid_to_annot[raw_id] = line.strip()

        f_bleu_eval_ref = open(dataset.name + '.ref', 'w')
        f_bleu_eval_hyp = open(dataset.name + '.hyp', 'w')

        logging.info('evaluating [%s] set, [%d] examples', dataset.name, dataset.count)

    cum_oracle_bleu = 0.0
    cum_oracle_acc = 0.0
    cum_bleu = 0.0
    cum_acc = 0.0
    sm = SmoothingFunction()

    all_references = []
    all_predictions = []

    if all(len(cand) == 0 for cand in decode_results):
        logging.ERROR('Empty decoding results for the current dataset!')
        return -1, -1

    for eid in range(dataset.count):
        example = dataset.examples[eid]
        ref_code = example.code
        ref_ast_tree = ast.parse(ref_code).body[0]
        refer_source = astor.to_source(ref_ast_tree).strip()
        # refer_source = ref_code
        refer_tokens = tokenize_code(refer_source)

        decode_cands = decode_results[eid]
        if len(decode_cands) == 0:
            continue

        decode_cand = decode_cands[0]

        cid, cand, ast_tree, code = decode_cand
        code = astor.to_source(ast_tree).strip()

        # simple_url_2_re = re.compile('_STR:0_', re.))
        try:
            predict_tokens = tokenize_code(code)
        except:
            logging.error('error in tokenizing [%s]', code)
            continue

        if refer_tokens == predict_tokens:
            cum_acc += 1
            pass

            if verbose:
                exact_match_ids.append(example.raw_id)
                f.write('-' * 60 + '\n')
                f.write('example_id: %d\n' % example.raw_id)
                f.write(code + '\n')
                f.write('-' * 60 + '\n')

        if MODE == 'django':
            ref_code_for_bleu = example.meta_data['raw_code']
            pred_code_for_bleu = de_canonicalize_code(code, example.meta_data['raw_code'])
            # ref_code_for_bleu = de_canonicalize_code(ref_code_for_bleu, example.meta_data['raw_code'])
            # convert canonicalized code to raw code
            for literal, place_holder in example.meta_data['str_map'].iteritems():
                pred_code_for_bleu = pred_code_for_bleu.replace('\'' + place_holder + '\'', literal)
                # ref_code_for_bleu = ref_code_for_bleu.replace('\'' + place_holder + '\'', literal)
        elif MODE == 'hs':
            ref_code_for_bleu = ref_code
            pred_code_for_bleu = code

        # we apply Ling Wang's trick when evaluating BLEU scores
        refer_tokens_for_bleu = tokenize_for_bleu_eval(ref_code_for_bleu)
        pred_tokens_for_bleu = tokenize_for_bleu_eval(pred_code_for_bleu)

        weired = False
        if refer_tokens_for_bleu == pred_tokens_for_bleu:
            # cum_acc += 1
            pass
        elif refer_tokens == predict_tokens:
            # weired!
            weired = True

        shorter = len(pred_tokens_for_bleu) < len(refer_tokens_for_bleu)

        all_references.append([refer_tokens_for_bleu])
        all_predictions.append(pred_tokens_for_bleu)

        # try:
        ngram_weights = [0.25] * min(4, len(refer_tokens_for_bleu))
        bleu_score = sentence_bleu([refer_tokens_for_bleu], pred_tokens_for_bleu, weights=ngram_weights, smoothing_function=sm.method3)
        cum_bleu += bleu_score
        # except:
        #    pass

        if verbose:
            print 'raw_id: %d, bleu_score: %f' % (example.raw_id, bleu_score)

            f_decode.write('-' * 60 + '\n')
            f_decode.write('example_id: %d\n' % example.raw_id)
            f_decode.write('intent: \n')

            if MODE == 'django':
                f_decode.write(eid_to_annot[example.raw_id] + '\n')
            elif MODE == 'hs':
                f_decode.write(' '.join(example.query) + '\n')

            f_bleu_eval_ref.write(' '.join(refer_tokens_for_bleu) + '\n')
            f_bleu_eval_hyp.write(' '.join(pred_tokens_for_bleu) + '\n')

            f_decode.write('canonicalized reference: \n')
            f_decode.write(refer_source + '\n')
            f_decode.write('canonicalized prediction: \n')
            f_decode.write(code + '\n')
            f_decode.write('reference code for bleu calculation: \n')
            f_decode.write(ref_code_for_bleu + '\n')
            f_decode.write('predicted code for bleu calculation: \n')
            f_decode.write(pred_code_for_bleu + '\n')
            f_decode.write('pred_shorter_than_ref: %s\n' % shorter)
            f_decode.write('weired: %s\n' % weired)
            f_decode.write('-' * 60 + '\n')


        # compute oracle
        best_score = -1000
        cur_oracle_acc = 0.
        for decode_cand in decode_cands[:10]:
            cid, cand, ast_tree, code = decode_cand
            try:
                predict_tokens = tokenize_code(code)
            except:
                continue

            if predict_tokens == refer_tokens:
                cur_oracle_acc = 1

            try:
                pred_tokens_for_bleu = tokenize_for_bleu_eval(code)
                bleu_score = sentence_bleu([refer_tokens_for_bleu], pred_tokens_for_bleu, smoothing_function=sm.method3)
            except:
                pass
                # print "Exception:"
                # print '-' * 60
                # print predict_tokens
                # print refer_tokens
                # traceback.print_exc(file=sys.stdout)
                # print '-' * 60

            if bleu_score > best_score:
                best_score = bleu_score

        cum_oracle_bleu += best_score
        cum_oracle_acc += cur_oracle_acc

    cum_bleu /= dataset.count
    cum_acc /= dataset.count
    cum_oracle_bleu /= dataset.count
    cum_oracle_acc /= dataset.count

    print 'corpus level bleu: %f' % corpus_bleu(all_references, all_predictions, smoothing_function=sm.method3)
    print 'sentence level bleu: %f' % cum_bleu
    print 'accuracy: %f' % cum_acc
    print 'oracle bleu: %f' % cum_oracle_bleu
    print 'oracle accuracy: %f' % cum_oracle_acc

    if verbose:
        f.write(', '.join(str(i) for i in exact_match_ids))
        f.close()
        f_decode.close()

        f_bleu_eval_ref.close()
        f_bleu_eval_hyp.close()

    return cum_bleu, cum_acc


def evaluate_ifttt_results(dataset, decode_results, verbose=True):
    assert dataset.count == len(decode_results)

    f = f_decode = None
    if verbose:
        f = open(dataset.name + '.exact_match', 'w')
        exact_match_ids = []
        f_decode = open(dataset.name + '.decode_results.txt', 'w')

        logging.info('evaluating [%s] set, [%d] examples', dataset.name, dataset.count)

    cum_channel_acc = 0.0
    cum_channel_func_acc = 0.0
    cum_prod_f1 = 0.0
    cum_oracle_prod_f1 = 0.0

    if all(len(cand) == 0 for cand in decode_results):
        logging.ERROR('Empty decoding results for the current dataset!')
        return -1, -1, -1

    for eid in range(dataset.count):
        example = dataset.examples[eid]
        ref_parse_tree = example.parse_tree
        decode_candidates = decode_results[eid]

        if len(decode_candidates) == 0:
            continue

        decode_cand = decode_candidates[0]

        cid, cand_hyp = decode_cand
        predict_parse_tree = cand_hyp.tree

        exact_match = predict_parse_tree == ref_parse_tree

        channel_acc, channel_func_acc, prod_f1 = ifttt_metric(predict_parse_tree, ref_parse_tree)
        cum_channel_acc += channel_acc
        cum_channel_func_acc += channel_func_acc
        cum_prod_f1 += prod_f1

        if verbose:
            if exact_match:
                exact_match_ids.append(example.raw_id)

            print 'raw_id: %d, prod_f1: %f' % (example.raw_id, prod_f1)

            f_decode.write('-' * 60 + '\n')
            f_decode.write('example_id: %d\n' % example.raw_id)
            f_decode.write('intent: \n')

            f_decode.write(' '.join(example.query) + '\n')

            f_decode.write('reference: \n')
            f_decode.write(str(ref_parse_tree) + '\n')
            f_decode.write('prediction: \n')
            f_decode.write(str(predict_parse_tree) + '\n')
            f_decode.write('-' * 60 + '\n')

        # compute oracle
        best_prod_f1 = -1.
        for decode_cand in decode_candidates[:10]:
            cid, cand_hyp = decode_cand
            predict_parse_tree = cand_hyp.tree

            channel_acc, channel_func_acc, prod_f1 = ifttt_metric(predict_parse_tree, ref_parse_tree)

            if prod_f1 > best_prod_f1:
                best_prod_f1 = prod_f1

        cum_oracle_prod_f1 += best_prod_f1

    cum_channel_acc /= dataset.count
    cum_channel_func_acc /= dataset.count
    cum_prod_f1 /= dataset.count
    cum_oracle_prod_f1 /= dataset.count

    logging.info('channel_acc: %f', cum_channel_acc)
    logging.info('channel_func_acc: %f', cum_channel_func_acc)
    logging.info('prod_f1: %f', cum_prod_f1)
    logging.info('oracle prod_f1: %f', cum_oracle_prod_f1)

    if verbose:
        f.write(', '.join(str(i) for i in exact_match_ids))
        f.close()
        f_decode.close()

    return cum_channel_acc, cum_channel_func_acc, cum_prod_f1


def ifttt_metric(predict_parse_tree, ref_parse_tree):
    channel_acc = channel_func_acc = prod_f1 = 0.
    # channel acc.
    channel_match = False
    if predict_parse_tree['TRIGGER'].children[0].type == ref_parse_tree['TRIGGER'].children[0].type and \
                    predict_parse_tree['ACTION'].children[0].type == ref_parse_tree['ACTION'].children[0].type:
        channel_acc += 1.
        channel_match = True

    # channel+func acc.
    if channel_match and predict_parse_tree['TRIGGER'].children[0].children[0].type == ref_parse_tree['TRIGGER'].children[0].children[0].type and \
                    predict_parse_tree['ACTION'].children[0].children[0].type == ref_parse_tree['ACTION'].children[0].children[0].type:
        channel_func_acc += 1.

    # prod. F1
    ref_rules, _ = ref_parse_tree.get_productions()
    predict_rules, _ = predict_parse_tree.get_productions()

    prod_f1 = len(set(ref_rules).intersection(set(predict_rules))) / len(ref_rules)

    return channel_acc, channel_func_acc, prod_f1


def decode_and_evaluate_ifttt(model, test_data):
    raw_ids = [int(i.strip()) for i in open('data/ifff.test_data.gold.id')]
    eids  = [i for i, e in enumerate(test_data.examples) if e.raw_id in raw_ids]
    test_data_subset = test_data.get_dataset_by_ids(eids, test_data.name + '.subset')

    from decoder import decode_ifttt_dataset
    decode_results = decode_ifttt_dataset(model, test_data_subset, verbose=True)
    evaluate_ifttt_results(test_data_subset, decode_results)


if __name__ == '__main__':
    from dataset import DataEntry, DataSet, Vocab, Action
    init_logging('parser.log', logging.INFO)

    train_data, dev_data, test_data = deserialize_from_file('data/ifttt.freq3.bin')
    decoding_results = []
    for eid in range(test_data.count):
        example = test_data.examples[eid]
        decoding_results.append([(eid, example.parse_tree)])

    evaluate_ifttt_results(test_data, decoding_results, verbose=True)
