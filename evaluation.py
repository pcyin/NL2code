from __future__ import division

from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction
import logging
import traceback

from model import *


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
    from lang.py.parse import tokenize_code
    import ast
    assert dataset.count == len(decode_results)

    f = f_decode = None
    if verbose:
        exact_match_ids = []
        f = open(dataset.name + '.exact_match', 'w')
        exact_match_ids = []
        f_decode = open(dataset.name + '.decode_results.txt', 'w')
        eid_to_annot = dict()
        for raw_id, line in enumerate(open('/Users/yinpengcheng/Research/SemanticParsing/CodeGeneration/en-django/all.anno')):
            eid_to_annot[raw_id] = line.strip()
        logging.info('evaluating [%s] set, [%d] examples', dataset.name, dataset.count)

    cum_oracle_bleu = 0.0
    cum_oracle_acc = 0.0
    cum_bleu = 0.0
    cum_acc = 0.0
    sm = SmoothingFunction()

    all_references = []
    all_predictions = []

    for eid in range(dataset.count):
        example = dataset.examples[eid]
        ref_code = example.code
        ref_ast_tree = ast.parse(ref_code).body[0]
        refer_source = astor.to_source(ref_ast_tree)
        refer_tokens = tokenize_code(refer_source)

        decode_cands = decode_results[eid]
        if len(decode_cands) == 0:
            continue

        decode_cand = decode_cands[0]

        cid, cand, ast_tree, code = decode_cand

        # simple_url_2_re = re.compile('_STR:0_', re.))
        try:
            predict_tokens = tokenize_code(code)
        except:
            logging.error('error in tokenizing [%s]', code)
            continue

        if refer_tokens == predict_tokens:
            cum_acc += 1

            if verbose:
                exact_match_ids.append(example.raw_id)
                f.write('-' * 60 + '\n')
                f.write('example_id: %d\n' % example.raw_id)
                f.write(code + '\n')
                f.write('-' * 60 + '\n')

        if verbose:
            f_decode.write('-' * 60 + '\n')
            f_decode.write('example_id: %d\n' % example.raw_id)
            f_decode.write('intent: \n')
            f_decode.write(eid_to_annot[example.raw_id] + '\n')
            f_decode.write('reference: \n')
            f_decode.write(refer_source + '\n')
            f_decode.write('prediction: \n')
            f_decode.write(code + '\n')
            f_decode.write('-' * 60 + '\n')


        # if hasattr(example, 'code'):
        #     ref_code = example.code
        #     ref_code = astor.to_source(ast.parse(ref_code))
        #     unescaped_cand_code = unescape(code)
        #
        #     refer_tokens2 = tokenize(ref_code)
        #     unescaped_cand_tokens = tokenize(unescaped_cand_code)
        #
        #     if refer_tokens2 == unescaped_cand_tokens:
        #         cum_new_acc += 1

        all_references.append([refer_tokens])
        all_predictions.append(predict_tokens)

        score = sentence_bleu([refer_tokens], predict_tokens, smoothing_function=sm.method3)
        if verbose:
            print 'raw_id: %d, score: %f' % (example.raw_id, score)

        cum_bleu += score

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
                score = sentence_bleu([refer_tokens], predict_tokens, smoothing_function=sm.method3)
            except:
                print "Exception:"
                print '-' * 60
                print predict_tokens
                print refer_tokens
                traceback.print_exc(file=sys.stdout)
                print '-' * 60

            if score > best_score:
                best_score = score

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

    return cum_bleu, cum_acc