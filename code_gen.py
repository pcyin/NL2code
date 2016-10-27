import numpy as np

from model import *
from dataset import *
from learner import Learner
from evaluation import *

from nn.utils.generic_utils import init_logging
from nn.utils.io_utils import deserialize_from_file

if __name__ == '__main__':
    init_logging('parser.log')
    # np.random.seed(1231)
    # query_tokens = np.random.randint(100, size=(3, 7), dtype='int32')
    # rules = np.random.randint(100, size=(3, 9), dtype='int32')
    #
    # model = Model()
    # model.build()
    # print model.train_func(query_tokens, rules)

    train_data, dev_data, test_data = deserialize_from_file('django.typed_rull.bin')

    model = Model()
    model.build()
    model.load('model.epoch10.npz')

    # model.save('model')
    # model.load('model.npz')
    # learner = Learner(model, train_data=train_data)

    # learner.train()

    evaluate(model, test_data)

    # example = test_data.get_examples(10)
    # hyps, hyp_scores = model.decode(example)
    #
    # for rule in example.rules:
    #     print rule
    #
    # i = 0
    # for hyp_id in range(10):
    #     print 'hyp #%d, score %f' % (i, hyp_scores[hyp_id])
    #     rules = [test_data.grammar_id_to_rule[rid] for rid in hyps[hyp_id]]
    #     print '| '.join([r.__repr__() for r in rules])
    #
    #     i += 1
