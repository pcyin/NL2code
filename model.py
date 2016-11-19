import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np

from collections import OrderedDict
import logging
import copy
import heapq
import sys

from nn.layers.embeddings import Embedding
from nn.layers.core import Dense, Dropout, WordDropout
from nn.layers.recurrent import BiLSTM, LSTM #, CondAttLSTM
import nn.optimizers as optimizers
import nn.initializations as initializations
from nn.activations import softmax
from nn.utils.theano_utils import *

from config import *
from grammar import *
from parse import *
from tree import *
from util import is_numeric
from components import Hyp, PointerNet, CondAttLSTM

sys.setrecursionlimit(50000)

class Model:
    def __init__(self):
        # self.symbol_embedding = Embedding(RULE_NUM, EMBED_DIM, name='symbol_embed')

        self.query_embedding = Embedding(SOURCE_VOCAB_SIZE, EMBED_DIM, name='query_embed')

        self.query_encoder_lstm = LSTM(EMBED_DIM, QUERY_DIM, return_sequences=True,
                                       name='query_encoder_lstm')

        self.decoder_lstm = CondAttLSTM(RULE_EMBED_DIM, LSTM_STATE_DIM, QUERY_DIM, DECODER_ATT_HIDDEN_DIM,
                                        name='decoder_lstm')

        self.src_ptr_net = PointerNet()

        self.terminal_gen_softmax = Dense(LSTM_STATE_DIM, 2, activation='softmax', name='terminal_gen_softmax')

        self.rule_embedding_W = initializations.get('glorot_uniform')((RULE_NUM, LSTM_STATE_DIM), name='rule_embedding_W')
        self.rule_embedding_b = shared_zeros(RULE_NUM, name='rule_embedding_b')

        self.vocab_embedding_W = initializations.get('glorot_uniform')((TARGET_VOCAB_SIZE, LSTM_STATE_DIM), name='vocab_embedding_W')
        self.vocab_embedding_b = shared_zeros(TARGET_VOCAB_SIZE, name='vocab_embedding_b')

        # self.rule_encoder_lstm.params
        self.params = self.query_embedding.params + self.query_encoder_lstm.params + \
                      self.decoder_lstm.params + self.src_ptr_net.params + self.terminal_gen_softmax.params + \
                      [self.rule_embedding_W, self.rule_embedding_b, self.vocab_embedding_W, self.vocab_embedding_b]

        self.srng = RandomStreams()

    def build(self):
        # (batch_size, max_example_action_num, action_type)
        tgt_action_seq = ndim_itensor(3, 'tgt_action_seq')

        # (batch_size, max_example_action_num, action_type)
        tgt_action_seq_type = ndim_itensor(3, 'tgt_action_seq_type')

        # (batch_size, max_query_length)
        query_tokens = ndim_itensor(2, 'query_tokens')

        # (batch_size, max_query_length, query_token_embed_dim)
        # (batch_size, max_query_length)
        query_token_embed, query_token_embed_mask = self.query_embedding(query_tokens, mask_zero=True)

        if WORD_DROPOUT > 0:
            logging.info('used word dropout for source, p = %f', WORD_DROPOUT)
            query_token_embed, query_token_embed_intact = WordDropout(WORD_DROPOUT, self.srng)(query_token_embed, False)

        batch_size = tgt_action_seq.shape[0]
        max_example_action_num = tgt_action_seq.shape[1]

        # action embeddings
        # (batch_size, max_example_action_num, action_embed_dim)
        tgt_action_seq_embed = T.switch(T.shape_padright(tgt_action_seq[:, :, 0] > 0),
                                        self.rule_embedding_W[tgt_action_seq[:, :, 0]],
                                        self.vocab_embedding_W[tgt_action_seq[:, :, 1]])

        tgt_action_seq_embed_tm1 = tensor_right_shift(tgt_action_seq_embed)

        # (batch_size, max_query_length, query_embed_dim)
        query_embed = self.query_encoder_lstm(query_token_embed, mask=query_token_embed_mask,
                                              dropout=DECODER_DROPOUT, srng=self.srng)
        
        # (batch_size, max_example_action_num, lstm_hidden_state)
        decoder_hidden_states, _, ctx_vectors = self.decoder_lstm(tgt_action_seq_embed_tm1,
                                                                  context=query_embed,
                                                                  context_mask=query_token_embed_mask,
                                                                  dropout=DECODER_DROPOUT,
                                                                  srng=self.srng,
                                                                  timestep=MAX_EXAMPLE_ACTION_NUM)

        # if DECODER_DROPOUT > 0:
        #     logging.info('used dropout for decoder output, p = %f', DECODER_DROPOUT)
        #     decoder_hidden_states = Dropout(DECODER_DROPOUT, self.srng)(decoder_hidden_states)

        # (batch_size, max_example_action_num, rule_num)
        rule_predict = softmax(T.dot(decoder_hidden_states, T.transpose(self.rule_embedding_W)) + self.rule_embedding_b)

        # (batch_size, max_example_action_num, 2)
        terminal_gen_action_prob = self.terminal_gen_softmax(decoder_hidden_states)

        # (batch_size, max_example_action_num, target_vocab_size)
        vocab_predict = softmax(T.dot(decoder_hidden_states, T.transpose(self.vocab_embedding_W)) + self.vocab_embedding_b)

        # (batch_size, max_example_action_num, max_query_length)
        copy_prob = self.src_ptr_net(query_embed, query_token_embed_mask, decoder_hidden_states)

        # (batch_size, max_example_action_num)
        tgt_action_seq_mask = T.any(tgt_action_seq_type, axis=-1)

        # (batch_size, max_example_action_num)
        rule_tgt_prob = rule_predict[T.shape_padright(T.arange(batch_size)),
                                     T.shape_padleft(T.arange(max_example_action_num)),
                                     tgt_action_seq[:, :, 0]]

        # (batch_size, max_example_action_num)
        vocab_tgt_prob = vocab_predict[T.shape_padright(T.arange(batch_size)),
                                       T.shape_padleft(T.arange(max_example_action_num)),
                                       tgt_action_seq[:, :, 1]]

        # (batch_size, max_example_action_num)
        copy_tgt_prob = copy_prob[T.shape_padright(T.arange(batch_size)),
                                  T.shape_padleft(T.arange(max_example_action_num)),
                                  tgt_action_seq[:, :, 2]]

        # (batch_size, max_example_action_num)
        tgt_prob = tgt_action_seq_type[:, :, 0] * rule_tgt_prob + \
                   tgt_action_seq_type[:, :, 1] * terminal_gen_action_prob[:, :, 0] * vocab_tgt_prob + \
                   tgt_action_seq_type[:, :, 2] * terminal_gen_action_prob[:, :, 1] * copy_tgt_prob

        likelihood = T.log(tgt_prob + 1.e-7 * (1 - tgt_action_seq_mask))
        loss = - (likelihood * tgt_action_seq_mask).sum(axis=-1) / tgt_action_seq_mask.sum(axis=-1)
        loss = T.mean(loss)

        # let's build the function!
        train_inputs = [query_tokens, tgt_action_seq, tgt_action_seq_type]
        optimizer = optimizers.get('adam')
        updates, grads = optimizer.get_updates(self.params, loss)
        self.train_func = theano.function(train_inputs, [loss],
                                          # [loss, tgt_action_seq_type, tgt_action_seq,
                                          #  rule_tgt_prob, vocab_tgt_prob, copy_tgt_prob,
                                          #  copy_prob, terminal_gen_action_prob],
                                          updates=updates)

        if WORD_DROPOUT > 0:
            self.build_decoder(query_tokens, query_token_embed_intact, query_token_embed_mask)
        else:
            self.build_decoder(query_tokens, query_token_embed, query_token_embed_mask)

    def build_decoder(self, query_tokens, query_token_embed, query_token_embed_mask):
        logging.info('building decoder ...')

        # (batch_size, decoder_state_dim)
        decoder_prev_state = ndim_tensor(2, name='decoder_prev_state')

        # (batch_size, decoder_state_dim)
        decoder_prev_cell = ndim_tensor(2, name='decoder_prev_cell')

        # (batch_size, decoder_state_dim)
        prev_action_embed = ndim_tensor(2, name='prev_action_embed')

        query_embed = self.query_encoder_lstm(query_token_embed, mask=query_token_embed_mask,
                                              dropout=DECODER_DROPOUT, train=False)

        # (batch_size, 1, decoder_state_dim)
        prev_action_embed_reshaped = prev_action_embed.dimshuffle((0, 'x', 1))

        # (batch_size, 1, decoder_state_dim)
        # (batch_size, 1, decoder_state_dim)
        # (batch_size, 1, field_token_encode_dim)
        decoder_next_state_dim3, decoder_next_cell_dim3, ctx_vectors = self.decoder_lstm(prev_action_embed_reshaped,
                                                                                         init_state=decoder_prev_state,
                                                                                         init_cell=decoder_prev_cell,
                                                                                         context=query_embed,
                                                                                         context_mask=query_token_embed_mask,
                                                                                         dropout=DECODER_DROPOUT,
                                                                                         train=False,
                                                                                         timestep=1)

        decoder_next_state = decoder_next_state_dim3.flatten(2)
        # decoder_output = decoder_next_state * (1 - DECODER_DROPOUT)

        decoder_next_cell = decoder_next_cell_dim3.flatten(2)

        rule_prob = softmax(T.dot(decoder_next_state, T.transpose(self.rule_embedding_W)) + self.rule_embedding_b)

        gen_action_prob = self.terminal_gen_softmax(decoder_next_state)

        vocab_prob = softmax(T.dot(decoder_next_state, T.transpose(self.vocab_embedding_W)) + self.vocab_embedding_b)

        copy_prob = self.src_ptr_net(query_embed, query_token_embed_mask, decoder_next_state_dim3)

        copy_prob = copy_prob.flatten(2)

        inputs = [query_tokens]
        outputs = [query_embed, query_token_embed_mask]

        self.decoder_func_init = theano.function(inputs, outputs)

        inputs = [decoder_prev_state, decoder_prev_cell, prev_action_embed,
                  query_embed, query_token_embed_mask]

        outputs = [decoder_next_state, decoder_next_cell,
                   rule_prob, gen_action_prob, vocab_prob, copy_prob]

        self.decoder_func_next_step = theano.function(inputs, outputs)

    def decode(self, example, grammar, terminal_vocab, beam_size, max_time_step):
        # beam search decoding

        eos = 1
        unk = terminal_vocab.unk
        vocab_embedding = self.vocab_embedding_W.get_value(borrow=True)
        rule_embedding = self.rule_embedding_W.get_value(borrow=True)

        query_tokens, _, _ = example.data

        query_embed, query_token_embed_mask = self.decoder_func_init(query_tokens)

        completed_hyps = []
        completed_hyp_num = 0
        live_hyp_num = 1

        root_hyp = Hyp()
        root_hyp.state = np.zeros(LSTM_STATE_DIM).astype('float32')
        root_hyp.cell = np.zeros(LSTM_STATE_DIM).astype('float32')
        root_hyp.action_embed = np.zeros(LSTM_STATE_DIM).astype('float32')

        hyp_samples = [root_hyp]  # [list() for i in range(live_hyp_num)]

        # source word id in the terminal vocab
        src_token_id = [terminal_vocab[t] for t in example.query][:MAX_QUERY_LENGTH]
        unk_pos_list = [x for x, t in enumerate(src_token_id) if t == unk]

        # sometimes a word may appear multi-times in the source, in this case,
        # we just copy its first appearing position. Therefore we mask the words
        # appearing second and onwards to -1
        token_set = set()
        for i, tid in enumerate(src_token_id):
            if tid in token_set:
                src_token_id[i] = -1
            else: token_set.add(tid)

        for t in xrange(max_time_step):
            # print 'time step [%d]' % t
            decoder_prev_state = np.array([hyp.state for hyp in hyp_samples]).astype('float32')
            decoder_prev_cell = np.array([hyp.cell for hyp in hyp_samples]).astype('float32')
            prev_action_embed = np.array([hyp.action_embed for hyp in hyp_samples]).astype('float32')

            query_embed_tiled = np.tile(query_embed, [live_hyp_num, 1, 1])
            query_token_embed_mask_tiled = np.tile(query_token_embed_mask, [live_hyp_num, 1])

            inputs = [decoder_prev_state, decoder_prev_cell, prev_action_embed,
                      query_embed_tiled, query_token_embed_mask_tiled]

            decoder_next_state, decoder_next_cell, \
            rule_prob, gen_action_prob, vocab_prob, copy_prob  = self.decoder_func_next_step(*inputs)

            new_hyp_samples = []

            cut_off_k = beam_size
            score_heap = []

            # iterating over items in the beam
            # print 'time step: %d, hyp num: %d' % (t, live_hyp_num)

            word_prob = gen_action_prob[:, 0:1] * vocab_prob
            word_prob[:, unk] = 0

            hyp_scores = np.array([hyp.score for hyp in hyp_samples])

            # word_prob[:, src_token_id] += gen_action_prob[:, 1:2] * copy_prob[:, :len(src_token_id)]
            # word_prob[:, unk] = 0

            rule_apply_cand_hyp_ids = []
            rule_apply_cand_scores = []
            rule_apply_cand_rules = []
            rule_apply_cand_rule_ids = []

            hyp_frontier_nts = []
            word_gen_hyp_ids = []

            unk_words = []

            for k in xrange(live_hyp_num):
                hyp = hyp_samples[k]

                # if k == 0:
                #     print 'Top Hyp: %s' % hyp.tree.__repr__()

                frontier_nt = hyp.frontier_nt()
                hyp_frontier_nts.append(frontier_nt)
                # we have a completed hyp
                if frontier_nt is None:
                    # hyp.score /= t
                    # hyp.score /= hyp.tree.size

                    # remove small-sized hyps
                    if t <= 2:
                        continue

                    hyp.n_timestep = t
                    completed_hyps.append(hyp)
                    completed_hyp_num += 1

                    continue

                # if it's not a leaf
                if not frontier_nt.holds_value:
                    # iterate over all the possible rules
                    rules = grammar[frontier_nt.node]
                    assert len(rules) > 0, 'fail to expand nt node %s' % frontier_nt
                    for rule in rules:
                        rule_id = grammar.rule_to_id[rule]

                        cur_rule_score = np.log(rule_prob[k, rule_id])
                        new_hyp_score = hyp.score + cur_rule_score

                        rule_apply_cand_hyp_ids.append(k)
                        rule_apply_cand_scores.append(new_hyp_score)
                        rule_apply_cand_rules.append(rule)
                        rule_apply_cand_rule_ids.append(rule_id)

                else:  # it's a leaf!
                    for i, tid in enumerate(src_token_id):
                        if tid != -1:
                            word_prob[k, tid] += gen_action_prob[k, 1] * copy_prob[k, i]

                    # and unk copy probability
                    if len(unk_pos_list) > 0:
                        unk_pos = copy_prob[k, unk_pos_list].argmax()
                        unk_pos = unk_pos_list[unk_pos]

                        unk_copy_score = gen_action_prob[k, 1] * copy_prob[k, unk_pos]
                        word_prob[k, unk] = unk_copy_score

                        unk_word = example.query[unk_pos]
                        unk_words.append(unk_word)

                    word_gen_hyp_ids.append(k)

            # prune the hyp space
            if completed_hyp_num >= beam_size:
                break

            word_prob = np.log(word_prob)

            word_gen_hyp_num = len(word_gen_hyp_ids)
            rule_apply_cand_num = len(rule_apply_cand_scores)

            if word_gen_hyp_num > 0:
                word_gen_cand_scores = hyp_scores[word_gen_hyp_ids, None] + word_prob[word_gen_hyp_ids, :]
                word_gen_cand_scores_flat = word_gen_cand_scores.flatten()

                cand_scores = np.concatenate([rule_apply_cand_scores, word_gen_cand_scores_flat])
            else:
                cand_scores = np.array(rule_apply_cand_scores)

            top_cand_ids = (-cand_scores).argsort()[:beam_size - completed_hyp_num]

            # expand_cand_num = 0
            for cand_id in top_cand_ids:
                # cand is rule application
                if cand_id < rule_apply_cand_num:
                    hyp_id = rule_apply_cand_hyp_ids[cand_id]
                    hyp = hyp_samples[hyp_id]
                    rule_id = rule_apply_cand_rule_ids[cand_id]
                    rule = rule_apply_cand_rules[cand_id]
                    new_hyp_score = rule_apply_cand_scores[cand_id]

                    new_tree = hyp.tree.copy()
                    new_hyp = Hyp(new_tree)
                    new_hyp.frontier_nt().apply_rule(rule)

                    new_hyp.score = new_hyp_score
                    new_hyp.state = copy.copy(decoder_next_state[hyp_id])
                    new_hyp.cell = copy.copy(decoder_next_cell[hyp_id])
                    new_hyp.action_embed = rule_embedding[rule_id]

                    new_hyp_samples.append(new_hyp)
                else:
                    tid = (cand_id - rule_apply_cand_num) % word_prob.shape[1]
                    word_gen_hyp_id = (cand_id - rule_apply_cand_num) / word_prob.shape[1]
                    hyp_id = word_gen_hyp_ids[word_gen_hyp_id]

                    if tid == unk:
                        token = unk_words[word_gen_hyp_id]
                    else:
                        token = terminal_vocab.id_token_map[tid]

                    # frontier_nt = hyp_frontier_nts[hyp_id]
                    # if frontier_nt.type == int and (not (is_numeric(token) or token == '<eos>')):
                    #     continue

                    hyp = hyp_samples[hyp_id]
                    new_hyp_score = word_gen_cand_scores[word_gen_hyp_id, tid]

                    new_tree = hyp.tree.copy()
                    new_hyp = Hyp(new_tree)
                    new_hyp.frontier_nt().append_token(token)

                    new_hyp.score = new_hyp_score
                    new_hyp.state = copy.copy(decoder_next_state[hyp_id])
                    new_hyp.cell = copy.copy(decoder_next_cell[hyp_id])
                    new_hyp.action_embed = vocab_embedding[tid]

                    new_hyp_samples.append(new_hyp)

                # expand_cand_num += 1
                # if expand_cand_num >= beam_size - completed_hyp_num:
                #     break

                # cand is word generation

            live_hyp_num = min(len(new_hyp_samples), beam_size - completed_hyp_num)
            if live_hyp_num < 1:
                break

            hyp_samples = new_hyp_samples
            # hyp_samples = sorted(new_hyp_samples, key=lambda x: x.score, reverse=True)[:live_hyp_num]

        completed_hyps = sorted(completed_hyps, key=lambda x: x.score, reverse=True)

        return completed_hyps

    @property
    def params_name_to_id(self):
        name_to_id = dict()
        for i, p in enumerate(self.params):
            assert p.name is not None
            # print 'parameter [%s]' % p.name

            name_to_id[p.name] = i

        return name_to_id

    @property
    def params_dict(self):
        assert len(set(p.name for p in self.params)) == len(self.params), 'param name clashes!'
        return OrderedDict((p.name, p) for p in self.params)

    def pull_params(self):
        return OrderedDict([(p_name, p.get_value(borrow=False)) for (p_name, p) in self.params_dict.iteritems()])

    def save(self, model_file, **kwargs):
        logging.info('save model to [%s]', model_file)

        weights_dict = self.pull_params()
        for k, v in kwargs.iteritems():
            weights_dict[k] = v

        np.savez(model_file, **weights_dict)

    def load(self, model_file):
        logging.info('load model from [%s]', model_file)
        weights_dict = np.load(model_file)

        # assert len(weights_dict.files) == len(self.params_dict)

        for p_name, p in self.params_dict.iteritems():
            if p_name not in weights_dict:
                logging.error('parameter [%s] not in saved weights file', p_name)
                return 1
            else:
                logging.info('loading parameter [%s]', p_name)
                assert np.array_equal(p.shape.eval(), weights_dict[p_name].shape), \
                    'shape mis-match for [%s]!, %s != %s' % (p_name, p.shape.eval(), weights_dict[p_name].shape)

                p.set_value(weights_dict[p_name])
