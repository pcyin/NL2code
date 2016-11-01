import nltk
from collections import OrderedDict, defaultdict
import logging
import collections
import numpy as np
import string
import re

from nn.utils.io_utils import serialize_to_file

from config import *


class Vocab(object):
    def __init__(self):
        self.token_id_map = OrderedDict()
        self.insert_token('<pad>')
        self.insert_token('<UNK>')

    def __getitem__(self, item):
        if item in self.token_id_map:
            return self.token_id_map[item]

        logging.warning('encounter one unknown word [%s]' % item)
        return self.token_id_map['<UNK>']

    def __setitem__(self, key, value):
        self.token_id_map[key] = value

    def __len__(self):
        return len(self.token_id_map)

    def __iter__(self):
        return self.token_id_map.iterkeys()

    def iteritems(self):
        return self.token_id_map.iteritems()

    def complete(self):
        self.id_token_map = dict((v, k) for (k, v) in self.token_id_map.iteritems())

    def get_token(self, token_id):
        return self.id_token_map[token_id]

    def insert_token(self, token):
        if token in self.token_id_map:
            return self[token]
        else:
            idx = len(self)
            self[token] = idx

            return idx


replace_punctuation = string.maketrans(string.punctuation, ' '*len(string.punctuation))


def tokenize(str):
    str = str.translate(replace_punctuation)
    return nltk.word_tokenize(str)


def gen_vocab(annot_file, vocab_size=3000):
    word_freq = defaultdict(int)
    lines = []
    for line in open(annot_file):
        line = line.strip()
        tokens = tokenize(line)
        lines.append(tokens)
        for token in tokens:
            word_freq[token] += 1

    print 'total num. of tokens: %d' % len(word_freq)

    ranked_words = sorted(word_freq, key=word_freq.get, reverse=True)[:vocab_size-2]
    ranked_words = set(ranked_words)
    vocab = Vocab()
    for tokens in lines:
        for token in tokens:
            if token in ranked_words:
                vocab.insert_token(token)

    return vocab


class DataEntry:
    def __init__(self, query, parse_tree):
        self.eid = -1
        self.query = query
        self.parse_tree = parse_tree
        self.rules = parse_tree.get_rule_list(include_leaf=True)

    @property
    def data(self):
        if not hasattr(self, '_data'):
            assert self.dataset is not None, 'No associated dataset for the example'

            self._data = self.dataset.get_prob_func_inputs([self.eid])

        return self._data


class DataSet:
    def __init__(self, vocab, grammar, name='train_data'):
        self.vocab = vocab
        self.name = name
        self.examples = []
        self.data_matrix = dict()
        self.grammar = grammar

    def add(self, example):
        example.eid = len(self.examples)
        self.examples.append(example)

    def get_dataset_by_ids(self, ids, name):
        dataset = DataSet(self.vocab, self.grammar, name)
        for eid in ids:
            dataset.add(self.examples[eid])

        for k, v in self.data_matrix.iteritems():
            dataset.data_matrix[k] = v[ids]

        return dataset

    @property
    def count(self):
        if self.examples:
            return len(self.examples)

        return 0

    def get_examples(self, ids):
        if isinstance(ids, collections.Iterable):
            return [self.examples[i] for i in ids]
        else:
            return self.examples[ids]

    def get_prob_func_inputs(self, ids):
        order = ['query_tokens', 'rules']

        return [self.data_matrix[x][ids] for x in order]

    def init_data_matrices(self):
        logging.info('init data matrices for [%s] dataset', self.name)
        vocab = self.vocab

        # np.max([len(e.query) for e in self.examples])
        # np.max([len(e.rules) for e in self.examples])

        query_tokens = self.data_matrix['query_tokens'] = np.zeros((self.count, MAX_QUERY_LENGTH), dtype='int32')
        rules = self.data_matrix['rules'] = np.zeros((self.count, MAX_EXAMPLE_RULE_NUM), dtype='int32')

        for eid, example in enumerate(self.examples):
            for tid, token in enumerate(example.query[:MAX_QUERY_LENGTH]):
                token_id = vocab[token]

                query_tokens[eid, tid] = token_id

            rule_num = len(example.rules[:MAX_EXAMPLE_RULE_NUM - 1])
            for rid, rule in enumerate(example.rules[:MAX_EXAMPLE_RULE_NUM - 1]):
                rules[eid, rid] = self.grammar.rule_to_id[rule]

            # end of rules
            rules[eid, rule_num] = 1

            example.dataset = self


def parse_django_dataset():
    from parse import parse_django

    annot_file = '/Users/yinpengcheng/Research/SemanticParsing/CodeGeneration/en-django/all.anno'

    vocab = gen_vocab(annot_file, vocab_size=4500)

    code_file = '/Users/yinpengcheng/Research/SemanticParsing/CodeGeneration/en-django/all.code'

    grammar, all_parse_trees = parse_django(code_file)

    train_data = DataSet(vocab, grammar, name='train')
    dev_data = DataSet(vocab, grammar, name='dev')
    test_data = DataSet(vocab, grammar, name='test')

    # train_data

    train_annot_file = '/Users/yinpengcheng/Research/SemanticParsing/CodeGeneration/en-django/train.anno'
    train_parse_trees = all_parse_trees[0:16000]
    for line, parse_tree in zip(open(train_annot_file), train_parse_trees):
        if parse_tree.is_leaf:
            continue

        line = line.strip()
        tokens = tokenize(line)
        entry = DataEntry(tokens, parse_tree)

        train_data.add(entry)

    train_data.init_data_matrices()

    # dev_data

    dev_annot_file = '/Users/yinpengcheng/Research/SemanticParsing/CodeGeneration/en-django/dev.anno'
    dev_parse_trees = all_parse_trees[16000:17000]
    for line, parse_tree in zip(open(dev_annot_file), dev_parse_trees):
        if parse_tree.is_leaf:
            continue

        line = line.strip()
        tokens = tokenize(line)
        entry = DataEntry(tokens, parse_tree)

        dev_data.add(entry)

    dev_data.init_data_matrices()

    # test_data

    test_annot_file = '/Users/yinpengcheng/Research/SemanticParsing/CodeGeneration/en-django/test.anno'
    test_parse_trees = all_parse_trees[17000:18805]
    for line, parse_tree in zip(open(test_annot_file), test_parse_trees):
        if parse_tree.is_leaf:
            continue

        line = line.strip()
        tokens = tokenize(line)
        entry = DataEntry(tokens, parse_tree)

        test_data.add(entry)

    test_data.init_data_matrices()

    serialize_to_file((train_data, dev_data, test_data), 'django.typed_rule.bin')


def check_terminals():
    from parse import parse_django, unescape
    grammar, parse_trees = parse_django('/Users/yinpengcheng/Research/SemanticParsing/CodeGeneration/en-django/all.code')
    annot_file = '/Users/yinpengcheng/Research/SemanticParsing/CodeGeneration/en-django/all.anno'

    unique_terminals = set()
    invalid_terminals = set()

    for i, line in enumerate(open(annot_file)):
        parse_tree = parse_trees[i]
        utterance = line.strip()

        leaves = parse_tree.get_leaves()
        # tokens = set(nltk.word_tokenize(utterance))
        leave_tokens = [l.label for l in leaves if l.label]

        not_included = []
        for leaf_token in leave_tokens:
            leaf_token = str(leaf_token)
            leaf_token = unescape(leaf_token)
            if leaf_token not in utterance:
                not_included.append(leaf_token)

                if len(leaf_token) <= 15:
                    unique_terminals.add(leaf_token)
                else:
                    invalid_terminals.add(leaf_token)
            else:
                if isinstance(leaf_token, str):
                    print leaf_token

        # if not_included:
        #     print str(i) + '---' + ', '.join(not_included)

    # print 'num of unique leaves: %d' % len(unique_terminals)
    # print unique_terminals
    #
    # print 'num of invalid leaves: %d' % len(invalid_terminals)
    # print invalid_terminals

QUOTED_STRING_RE = re.compile(r"(?P<quote>['\"])(?P<string>.*?)(?<!\\)(?P=quote)")


def process_query(query):
    str_count = 0
    while QUOTED_STRING_RE.search(query):
        query = QUOTED_STRING_RE.sub('{STR:%d}' % str_count, query, 1)
        str_count += 1

    return query

if __name__== '__main__':
    from nn.utils.generic_utils import init_logging
    init_logging('parse.log')

    # parse_django_dataset()
    check_terminals()

    # for i, query in enumerate(open('/Users/yinpengcheng/Research/SemanticParsing/CodeGeneration/en-django/all.anno')):
    #     print i, process_query(query)
