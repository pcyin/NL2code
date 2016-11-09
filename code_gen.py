import numpy as np
import cProfile
import ast
import traceback
import argparse
import os
from vprof import profiler

from model import Model
from dataset import DataEntry, DataSet, Vocab, Action
from learner import Learner
from evaluation import *
from decoder import decode_dataset
from parse import decode_tree_to_ast
from components import Hyp
from tree import Tree

from nn.utils.generic_utils import init_logging
from nn.utils.io_utils import deserialize_from_file, serialize_to_file

def parse_args():
    parser = argparse.ArgumentParser()
    sub_parsers = parser.add_subparsers(dest='operation')
    train_parser = sub_parsers.add_parser('train')
    test_parser = sub_parsers.add_parser('decode')
    interactive_parser = sub_parsers.add_parser('interactive')
    evaluate_parser = sub_parsers.add_parser('evaluate')

    parser.add_argument('-data')
    parser.add_argument('-model', default=None)
    parser.add_argument('-conf', default='config.py', help='config file name')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    init_logging('parser.log')
    args = parse_args()

    logging.info('current config: %s', config_info)

    # np.random.seed(1231)

    dataset_file = 'django.cleaned.dataset.bin'

    if args.data:
        dataset_file = args.data


    logging.info('loading dataset [%s]', dataset_file)
    train_data, dev_data, test_data = deserialize_from_file(dataset_file)

    if args.operation in ['train', 'decode', 'interactive']:
        model = Model()
        model.build()

        if args.model:
            model.load(args.model)

    if args.operation == 'decode':
        dataset = test_data.get_dataset_by_ids([1,2,3,4,5,6,7,8,9,10], name='sample')
        # cProfile.run('decode_dataset(model, dataset)', sort=2)
        decode_dataset(model, dataset)


    # profiler.run(model.decode, 'h', args=(example, train_data.grammar, train_data.terminal_vocab),
    #              kwargs={'beam_size': 30, 'max_time_step': 15},
    #              host='localhost', port=8000)


    if args.operation == 'evaluate':
        decode_results = deserialize_from_file('test_data.decode_results.epoch15')
        dataset = test_data

        evaluate_decode_results(dataset, decode_results)

    if args.operation == 'interactive':
        assert model is not None
        while True:
            example_id = raw_input('example id: ')
            try:
                example_id = int(example_id)
            except:
                continue
            # example_id = int(example_id.strip())

            example = [e for e in test_data.examples if e.raw_id == example_id][0]

            print 'gold parse tree:'
            print example.parse_tree

            cand_list = model.decode(example, train_data.grammar, train_data.terminal_vocab,
                                     beam_size=50, max_time_step=100)

            # serialize_to_file(cand_list, 'cand_hyps.%d.bin' % example.raw_id)

            from parse import decode_tree_to_ast

            for cid, cand in enumerate(cand_list[:10]):
                print '*' * 60
                print 'cand #%d, score: %f' % (cid, cand.score)

                try:
                    ast_tree = decode_tree_to_ast(cand.tree)
                    code = astor.to_source(ast_tree)
                    print 'code: ', code
                except:
                    print "Exception in decoding:"
                    print '-' * 60
                    traceback.print_exc(file=sys.stdout)
                    print '-' * 60
                finally:
                    print cand.tree.__repr__()
                    print '*' * 60
