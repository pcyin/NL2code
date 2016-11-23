from __future__ import division
import copy

import nltk
from collections import OrderedDict, defaultdict
import logging
import collections
import numpy as np
import string
import re
import astor
from itertools import chain

from nn.utils.io_utils import serialize_to_file, deserialize_from_file

from parse import tree_to_ast, ast_to_tree, parse, parse_django
from grammar import TypedRule
from config import *


class Action(object):
    def __init__(self, act_type, data):
        self.act_type = act_type
        self.data = data

    def __repr__(self):
        data_str = self.data if not isinstance(self.data, dict) else \
            ', '.join(['%s: %s' % (k, v) for k, v in self.data.iteritems()])
        repr_str = 'Action{%s}[%s]' % (ACTION_NAMES[self.act_type], data_str)

        return repr_str


class Vocab(object):
    def __init__(self):
        self.token_id_map = OrderedDict()
        self.insert_token('<pad>')
        self.insert_token('<unk>')
        self.insert_token('<eos>')

    @property
    def unk(self):
        return self.token_id_map['<unk>']

    @property
    def eos(self):
        return self.token_id_map['<eos>']

    def __getitem__(self, item):
        if item in self.token_id_map:
            return self.token_id_map[item]

        logging.debug('encounter one unknown word [%s]' % item)
        return self.token_id_map['<unk>']

    def __contains__(self, item):
        return item in self.token_id_map

    @property
    def size(self):
        return len(self.token_id_map)

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


def gen_vocab(tokens, vocab_size=3000, freq_cutoff=5):
    word_freq = defaultdict(int)

    for token in tokens:
        word_freq[token] += 1

    print 'total num. of tokens: %d' % len(word_freq)

    words_freq_cutoff = [w for w in word_freq if word_freq[w] >= freq_cutoff]
    print 'num. of words appear at least %d: %d' % (freq_cutoff, len(words_freq_cutoff))

    ranked_words = sorted(words_freq_cutoff, key=word_freq.get, reverse=True)[:vocab_size-2]
    ranked_words = set(ranked_words)

    vocab = Vocab()
    for token in tokens:
        if token in ranked_words:
            vocab.insert_token(token)

    vocab.complete()

    return vocab


class DataEntry:
    def __init__(self, raw_id, query, parse_tree, code, actions):
        self.raw_id = raw_id
        self.eid = -1
        self.query = query
        self.parse_tree = parse_tree
        self.actions = actions
        self.code = code

    @property
    def data(self):
        if not hasattr(self, '_data'):
            assert self.dataset is not None, 'No associated dataset for the example'

            self._data = self.dataset.get_prob_func_inputs([self.eid])

        return self._data

    def copy(self):
        e = DataEntry(self.raw_id, self.query, self.parse_tree, self.code, self.actions)

        return e


class DataSet:
    def __init__(self, annot_vocab, terminal_vocab, grammar, name='train_data'):
        self.annot_vocab = annot_vocab
        self.terminal_vocab = terminal_vocab
        self.name = name
        self.examples = []
        self.data_matrix = dict()
        self.grammar = grammar

    def add(self, example):
        example.eid = len(self.examples)
        example.dataset = self
        self.examples.append(example)

    def get_dataset_by_ids(self, ids, name):
        dataset = DataSet(self.annot_vocab, self.terminal_vocab,
                          self.grammar, name)
        for eid in ids:
            example_copy = self.examples[eid].copy()
            dataset.add(example_copy)

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
        order = ['query_tokens', 'tgt_action_seq', 'tgt_action_seq_type',
                 'tgt_node_seq', 'tgt_par_rule_seq', 'tgt_par_t_seq']

        max_src_seq_len = max(len(self.examples[i].query) for i in ids)
        max_tgt_seq_len = max(len(self.examples[i].actions) for i in ids)

        logging.debug('max. src sequence length: %d', max_src_seq_len)
        logging.debug('max. tgt sequence length: %d', max_tgt_seq_len)

        data = []
        for entry in order:
            if entry == 'query_tokens':
                data.append(self.data_matrix[entry][ids, :max_src_seq_len])
            else:
                data.append(self.data_matrix[entry][ids, :max_tgt_seq_len])

        return data


    def init_data_matrices(self):
        logging.info('init data matrices for [%s] dataset', self.name)
        annot_vocab = self.annot_vocab
        terminal_vocab = self.terminal_vocab

        # np.max([len(e.query) for e in self.examples])
        # np.max([len(e.rules) for e in self.examples])

        query_tokens = self.data_matrix['query_tokens'] = np.zeros((self.count, MAX_QUERY_LENGTH), dtype='int32')
        tgt_node_seq = self.data_matrix['tgt_node_seq'] = np.zeros((self.count, MAX_EXAMPLE_ACTION_NUM), dtype='int32')
        tgt_par_rule_seq = self.data_matrix['tgt_par_rule_seq'] = np.zeros((self.count, MAX_EXAMPLE_ACTION_NUM), dtype='int32')
        tgt_par_t_seq = self.data_matrix['tgt_par_t_seq'] = np.zeros((self.count, MAX_EXAMPLE_ACTION_NUM), dtype='int32')
        tgt_action_seq = self.data_matrix['tgt_action_seq'] = np.zeros((self.count, MAX_EXAMPLE_ACTION_NUM, 3), dtype='int32')
        tgt_action_seq_type = self.data_matrix['tgt_action_seq_type'] = np.zeros((self.count, MAX_EXAMPLE_ACTION_NUM, 3), dtype='int32')

        for eid, example in enumerate(self.examples):
            exg_query_tokens = example.query[:MAX_QUERY_LENGTH]
            exg_action_seq = example.actions[:MAX_EXAMPLE_ACTION_NUM]

            for tid, token in enumerate(exg_query_tokens):
                token_id = annot_vocab[token]

                query_tokens[eid, tid] = token_id

            assert len(exg_action_seq) > 0

            for t, action in enumerate(exg_action_seq):
                if action.act_type == APPLY_RULE:
                    rule = action.data['rule']
                    tgt_action_seq[eid, t, 0] = self.grammar.rule_to_id[rule]
                    tgt_action_seq_type[eid, t, 0] = 1
                elif action.act_type == GEN_TOKEN:
                    token = action.data['literal']
                    token_id = terminal_vocab[token]
                    tgt_action_seq[eid, t, 1] = token_id
                    tgt_action_seq_type[eid, t, 1] = 1
                elif action.act_type == COPY_TOKEN:
                    src_token_idx = action.data['source_idx']
                    tgt_action_seq[eid, t, 2] = src_token_idx
                    tgt_action_seq_type[eid, t, 2] = 1
                elif action.act_type == GEN_COPY_TOKEN:
                    token = action.data['literal']
                    token_id = terminal_vocab[token]
                    tgt_action_seq[eid, t, 1] = token_id
                    tgt_action_seq_type[eid, t, 1] = 1

                    src_token_idx = action.data['source_idx']
                    tgt_action_seq[eid, t, 2] = src_token_idx
                    tgt_action_seq_type[eid, t, 2] = 1
                else:
                    raise RuntimeError('wrong action type!')

                # parent information
                rule = action.data['rule']
                tree = rule.tree
                parent_tree = tree.parent

                if action.act_type == APPLY_RULE:
                    tgt_node_seq[eid, t] = self.grammar.get_node_type_id(rule.parent)
                    if parent_tree:
                        parent_rule = parent_tree.to_rule()
                        tgt_par_rule_seq[eid, t] = self.grammar.rule_to_id[parent_rule]
                    else:
                        assert t == 0
                        tgt_par_rule_seq[eid, t] = -1
                else:
                    tgt_par_rule_seq[eid, t] = self.grammar.rule_to_id[rule]
                    tgt_node_seq[eid, t] = self.grammar.get_node_type_id(rule.children[0].type)

                # parent hidden states
                parent_t = action.data['parent_t']
                tgt_par_t_seq[eid, t] = parent_t

            example.dataset = self


def parse_django_dataset_nt_only():
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


def parse_django_dataset():
    from grammar import is_builtin_type
    from parse import unescape

    annot_file = '/Users/yinpengcheng/Research/SemanticParsing/CodeGeneration/en-django/all.anno'
    code_file = '/Users/yinpengcheng/Research/SemanticParsing/CodeGeneration/en-django/all.code'

    data = preprocess_dataset(annot_file, code_file)

    annot_tokens = list(chain(*[e['query_tokens'] for e in data]))
    annot_vocab = gen_vocab(annot_tokens, vocab_size=5000, freq_cutoff=5) # gen_vocab(annot_tokens, vocab_size=5980)

    terminal_token_seq = []
    empty_actions_count = 0

    # helper function begins
    def get_terminal_tokens(_terminal_str):
        tmp_terminal_tokens = _terminal_str.split('-SP-')
        _terminal_tokens = []
        for token in tmp_terminal_tokens:
            if token:
                _terminal_tokens.append(token)
            _terminal_tokens.append('-SP-')

        return _terminal_tokens[:-1]
    # helper function ends

    # build grammar ...
    grammar, all_parse_trees = parse_django(code_file)

    # first pass
    for entry in data:
        idx = entry['id']
        query_tokens = entry['query_tokens']
        code = entry['code']

        parse_tree = parse(code)
        rule_list, rule_pos_list, par_rule_pos_list = parse_tree.get_rule_list(include_leaf=True, leaf_val=True)

        for rule in rule_list:
            if is_builtin_type(rule.parent.type):
                assert rule.parent.label is None
                assert len(rule.children) == 1
                terminal_val = rule.children[0].label

                terminal_str = str(terminal_val)
                # print idx, terminal_str
                terminal_tokens = get_terminal_tokens(terminal_str)

                for terminal_token in terminal_tokens:
                    assert len(terminal_token) > 0
                    terminal_token_seq.append(terminal_token)

    terminal_vocab = gen_vocab(terminal_token_seq, vocab_size=4830, freq_cutoff=5)

    train_data = DataSet(annot_vocab, terminal_vocab, grammar, 'train_data')
    dev_data = DataSet(annot_vocab, terminal_vocab, grammar, 'dev_data')
    test_data = DataSet(annot_vocab, terminal_vocab, grammar, 'test_data')

    all_examples = []

    can_fully_gen_num = 0

    # second pass
    for entry in data:
        idx = entry['id']
        query_tokens = entry['query_tokens']
        code = entry['code']
        str_map = entry['str_map']

        parse_tree = parse(code)
        rule_list, rule_pos_list, par_rule_pos_list = parse_tree.get_rule_list(include_leaf=True, leaf_val=True)

        actions = []

        can_fully_gen = True

        # import astor
        # from parse import code_to_ast
        # ref_code = astor.to_source(code_to_ast(code)).replace('\"','\'')
        # ref_code2 = unescape(astor.to_source(tree_to_ast(parse_tree)))
        #
        # if ref_code != ref_code2:
        #     print '*' * 60
        #     print idx
        #     print ref_code
        #     print ref_code2
        #     print '*' * 60

        gen_terminal_action_count = 0
        rule_pos_map = dict()
        for rule, rule_pos, par_rule_pos in zip(rule_list, rule_pos_list, par_rule_pos_list):
            if not is_builtin_type(rule.parent.type):
                rule_pos_map[rule_pos] = rule_pos + gen_terminal_action_count
                if len(actions) > 0:
                    parent_t = rule_pos_map[par_rule_pos]
                else:
                    parent_t = 0

                d = {'rule': rule, 'parent_t': parent_t}
                action = Action(APPLY_RULE, d)

                actions.append(action)
            if is_builtin_type(rule.parent.type):
                assert rule.parent.label is None
                assert len(rule.children) == 1

                # str -> str{val}
                # TODO: remove dummy rules?
                terminal_rule = copy.deepcopy(rule)
                terminal_rule.children[0].label = 'val'

                parent_t = rule_pos_map[par_rule_pos]
                d = {'rule': terminal_rule, 'parent_t': parent_t}

                actions.append(Action(APPLY_RULE, d))
                # time step for this terminal rule application
                terminal_rule_t = len(actions) - 1

                terminal_val = rule.children[0].label

                terminal_str = str(terminal_val)
                terminal_tokens = get_terminal_tokens(terminal_str)

                # print idx, terminal_str
                # terminal_tokens = [t for t in terminal_str.split(' ') if len(t) > 0]

                assert len(terminal_tokens) > 0

                for terminal_token in terminal_tokens:
                    term_tok_id = terminal_vocab[terminal_token]
                    tok_src_idx = -1
                    try:
                        tok_src_idx = query_tokens.index(terminal_token)
                    except ValueError:
                        pass

                    d = {'literal': terminal_token, 'rule': terminal_rule, 'parent_t': terminal_rule_t}

                    # cannot copy, only generation
                    # could be unk!
                    if tok_src_idx < 0 or tok_src_idx >= MAX_QUERY_LENGTH:
                        action = Action(GEN_TOKEN, d)
                        if terminal_token not in terminal_vocab:
                            if terminal_token not in query_tokens:
                                # print terminal_token
                                can_fully_gen = False
                    else:  # copy
                        if term_tok_id != terminal_vocab.unk:
                            d['source_idx'] = tok_src_idx
                            action = Action(GEN_COPY_TOKEN, d)
                        else:
                            d['source_idx'] = tok_src_idx
                            action = Action(COPY_TOKEN, d)

                    actions.append(action)
                    gen_terminal_action_count += 1

                d = {'literal': '<eos>', 'rule': terminal_rule, 'parent_t': terminal_rule_t}
                actions.append(Action(GEN_TOKEN, d))
                gen_terminal_action_count += 1

        if len(actions) == 0:
            empty_actions_count += 1
            continue

        example = DataEntry(idx, query_tokens, parse_tree, code, actions)

        if can_fully_gen:
            can_fully_gen_num += 1

        # train, valid, test
        if 0 <= idx < 16000:
            train_data.add(example)
        elif 16000 <= idx < 17000:
            dev_data.add(example)
        else:
            test_data.add(example)

        all_examples.append(example)

    # print statistics
    max_query_len = max(len(e.query) for e in all_examples)
    max_actions_len = max(len(e.actions) for e in all_examples)

    serialize_to_file([len(e.query) for e in all_examples], 'query.len')
    serialize_to_file([len(e.actions) for e in all_examples], 'actions.len')

    logging.info('examples that can be fully reconstructed: %d/%d=%f',
                 can_fully_gen_num, len(all_examples),
                 can_fully_gen_num / len(all_examples))
    logging.info('empty_actions_count: %d', empty_actions_count)
    logging.info('max_query_len: %d', max_query_len)
    logging.info('max_actions_len: %d', max_actions_len)

    train_data.init_data_matrices()
    dev_data.init_data_matrices()
    test_data.init_data_matrices()

    serialize_to_file((train_data, dev_data, test_data), 'data/django.cleaned.dataset.freq5.par_info.bin')

    return train_data, dev_data, test_data


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


def process_query(query, code):
    from parse import code_to_ast, ast_to_tree, tree_to_ast, parse
    import astor
    str_count = 0
    str_map = dict()

    match_count = 1
    match = QUOTED_STRING_RE.search(query)
    while match:
        str_repr = '_STR:%d_' % str_count
        str_literal = match.group(0)
        str_string = match.group(2)

        match_count += 1

        # if match_count > 50:
        #     return
        #

        query = QUOTED_STRING_RE.sub(str_repr, query, 1)
        str_map[str_literal] = str_repr

        str_count += 1
        match = QUOTED_STRING_RE.search(query)

        code = code.replace(str_literal, '\'' + str_repr + '\'')

    # clean the annotation
    # query = query.replace('.', ' . ')

    for k, v in str_map.iteritems():
        if k == '\'%s\'' or k == '\"%s\"':
            query = query.replace(v, k)
            code = code.replace('\'' + v + '\'', k)

    # tokenize
    query_tokens = nltk.word_tokenize(query)

    new_query_tokens = []
    # break up function calls
    for token in query_tokens:
        new_query_tokens.append(token)
        i = token.find('.')
        if 0 < i < len(token) - 1:
            new_tokens = ['['] + token.replace('.', ' . ').split(' ') + [']']
            new_query_tokens.extend(new_tokens)

    # check if the code compiles
    tree = parse(code)
    ast_tree = tree_to_ast(tree)
    astor.to_source(ast_tree)

    return new_query_tokens, code, str_map

def preprocess_dataset(annot_file, code_file):
    f_annot = open('annot.all.txt', 'w')
    f_code = open('code.all.txt', 'w')

    examples = []

    err_num = 0
    for idx, (annot, code) in enumerate(zip(open(annot_file), open(code_file))):
        annot = annot.strip()
        code = code.strip()
        try:
            clean_query_tokens, clean_code, str_map = process_query(annot, code)
            example = {'id': idx, 'query_tokens': clean_query_tokens, 'code': clean_code, 'str_map': str_map}
            examples.append(example)

            f_annot.write('example# %d\n' % idx)
            f_annot.write(' '.join(clean_query_tokens) + '\n')
            f_annot.write('%d\n' % len(str_map))
            for k, v in str_map.iteritems():
                f_annot.write('%s ||| %s\n' % (k, v))

            f_code.write('example# %d\n' % idx)
            f_code.write(clean_code + '\n')
        except:
            print code
            err_num += 1

        idx += 1

    f_annot.close()
    f_annot.close()

    serialize_to_file(examples, 'django.cleaned.bin')

    print 'error num: %d' % err_num
    print 'preprocess_dataset: cleaned example num: %d' % len(examples)

    return examples

if __name__== '__main__':
    from nn.utils.generic_utils import init_logging
    init_logging('parse.log')

    # parse_django_dataset()
    # check_terminals()

    # print process_query(""" ALLOWED_VARIABLE_CHARS is a string 'abcdefgh"ijklm" nop"%s"qrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.'.""")

    # for i, query in enumerate(open('/Users/yinpengcheng/Research/SemanticParsing/CodeGeneration/en-django/all.anno')):
    #     print i, process_query(query)

    # clean_dataset()

    parse_django_dataset()
