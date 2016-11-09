import theano
import theano.tensor as T
import numpy as np

from collections import OrderedDict
import logging
import copy
import heapq

from nn.layers.embeddings import Embedding
from nn.layers.core import Dense
from nn.layers.recurrent import BiLSTM, LSTM, CondAttLSTM
from nn.utils.theano_utils import ndim_itensor, tensor_right_shift, ndim_tensor
import nn.optimizers as optimizers
import nn.initializations as initializations
from nn.activations import softmax
from nn.utils.theano_utils import *

from config import *
from grammar import *
from parse import *
from tree import *
from components import Hyp, PointerNet


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

        batch_size = tgt_action_seq.shape[0]
        max_example_action_num = tgt_action_seq.shape[1]

        # action embeddings
        # (batch_size, max_example_action_num, action_embed_dim)
        tgt_action_seq_embed = T.switch(T.shape_padright(tgt_action_seq[:, :, 0] > 0),
                                        self.rule_embedding_W[tgt_action_seq[:, :, 0]],
                                        self.vocab_embedding_W[tgt_action_seq[:, :, 1]])

        tgt_action_seq_embed_tm1 = tensor_right_shift(tgt_action_seq_embed)

        # (batch_size, max_query_length, query_embed_dim)
        query_embed = self.query_encoder_lstm(query_token_embed, mask=query_token_embed_mask)
        
        # (batch_size, max_example_action_num, lstm_hidden_state)
        decoder_hidden_states, _, ctx_vectors = self.decoder_lstm(tgt_action_seq_embed_tm1,
                                                                  context=query_embed,
                                                                  context_mask=query_token_embed_mask)

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
        self.train_func = theano.function(train_inputs,
                                          [loss, tgt_action_seq_type, tgt_action_seq,
                                           rule_tgt_prob, vocab_tgt_prob, copy_tgt_prob,
                                           copy_prob, terminal_gen_action_prob],
                                          updates=updates)

        self.build_decoder(query_tokens, query_embed, query_token_embed_mask)

    def build_decoder(self, query_tokens, query_embed, query_token_embed_mask):
        logging.info('building decoder ...')

        # (batch_size, decoder_state_dim)
        decoder_prev_state = ndim_tensor(2, name='decoder_prev_state')

        # (batch_size, decoder_state_dim)
        decoder_prev_cell = ndim_tensor(2, name='decoder_prev_cell')

        # (batch_size, decoder_state_dim)
        prev_action_embed = ndim_tensor(2, name='prev_action_embed')

        # (batch_size, 1, decoder_state_dim)
        prev_action_embed_reshaped = prev_action_embed.dimshuffle((0, 'x', 1))

        # (batch_size, 1, decoder_state_dim)
        # (batch_size, 1, decoder_state_dim)
        # (batch_size, 1, field_token_encode_dim)
        decoder_next_state_dim3, decoder_next_cell_dim3, ctx_vectors = self.decoder_lstm(prev_action_embed_reshaped,
                                                                                         init_state=decoder_prev_state,
                                                                                         init_cell=decoder_prev_cell,
                                                                                         context=query_embed,
                                                                                         context_mask=query_token_embed_mask)

        decoder_next_state = decoder_next_state_dim3.flatten(2)
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

    def decode(self, example, grammar, terminal_vocab, beam_size=50, max_time_step=100):
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

        for t in range(max_time_step):
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
            for k in range(live_hyp_num):
                hyp = hyp_samples[k]

                # if k == 0:
                #     print 'Top Hyp: %s' % hyp.tree.__repr__()

                frontier_nt = hyp.frontier_nt()
                # we have a completed hyp
                if frontier_nt is None:
                    # hyp.score /= hyp.tree.size
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

                        if len(score_heap) == cut_off_k:
                            if score_heap[0] > new_hyp_score:
                                continue
                            else:
                                heapq.heappushpop(score_heap, new_hyp_score)
                        else:
                            heapq.heappush(score_heap, new_hyp_score)

                        new_tree = hyp.tree.copy() # copy.deepcopy(hyp.tree)
                        new_hyp = Hyp(new_tree)
                        new_hyp.frontier_nt().apply_rule(rule)

                        new_hyp.score = new_hyp_score
                        new_hyp.state = copy.copy(decoder_next_state[k])
                        new_hyp.cell = copy.copy(decoder_next_cell[k])
                        new_hyp.action_embed = rule_embedding[rule_id]

                        new_hyp_samples.append(new_hyp)
                else:  # it's a leaf!
                    for token, tid in terminal_vocab.iteritems():
                        tok_src_idx = -1
                        try:
                            tok_src_idx = example.query.index(token)
                        except ValueError: pass

                        # can only generate, should not be unk!
                        if tid != unk and (tok_src_idx < 0 or tok_src_idx >= MAX_QUERY_LENGTH):
                            p_gen = np.log(gen_action_prob[k, 0])
                            p_st = np.log(vocab_prob[k, tid])
                            score = p_gen + p_st
                        elif tid != unk:
                            # can both generate and copy
                            p_gen = np.log(gen_action_prob[k, 0])
                            p_st_gen = np.log(vocab_prob[k, tid])
                            p_copy = np.log(gen_action_prob[k, 1])
                            p_st_copy = np.log(copy_prob[k, tok_src_idx])
                            score = np.logaddexp(p_gen + p_st_gen, p_copy + p_st_copy)
                        else:
                            assert tid == unk
                            # it's a unk!
                            # can only copy
                            p_copy = np.log(gen_action_prob[k, 1])
                            # p_st_copy = np.log(copy_prob[k, tok_src_idx])
                            tok_src_idx = copy_prob[k].argmax()
                            token = example.query[tok_src_idx]

                            if token in terminal_vocab:
                                continue

                            p_st_copy = np.log(copy_prob[k, tok_src_idx])
                            score = p_copy + p_st_copy

                        new_hyp_score = hyp.score + score

                        if len(score_heap) == cut_off_k:
                            if score_heap[0] > new_hyp_score:
                                continue
                            else:
                                heapq.heappushpop(score_heap, new_hyp_score)
                        else:
                            heapq.heappush(score_heap, new_hyp_score)

                        new_tree = hyp.tree.copy()  # copy.deepcopy(hyp.tree)
                        new_hyp = Hyp(new_tree)
                        new_hyp.frontier_nt().append_token(token)

                        new_hyp.score = new_hyp_score
                        new_hyp.state = copy.copy(decoder_next_state[k])
                        new_hyp.cell = copy.copy(decoder_next_cell[k])
                        new_hyp.action_embed = vocab_embedding[tid]

                        new_hyp_samples.append(new_hyp)

            # prune the hyp space
            if completed_hyp_num >= beam_size:
                break

            live_hyp_num = min(len(new_hyp_samples), beam_size - completed_hyp_num)
            if live_hyp_num < 1:
                break

            hyp_samples = sorted(new_hyp_samples, key=lambda x: x.score, reverse=True)[:live_hyp_num]

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
        return OrderedDict([(p.name, p) for p in self.params])

    def save(self, model_file):
        logging.info('save model to [%s]', model_file)
        weights_dict = OrderedDict([(p_name, p.get_value()) for (p_name, p) in self.params_dict.iteritems()])
        np.savez(model_file, **weights_dict)

    def load(self, model_file):
        logging.info('load model from [%s]', model_file)
        weights_dict = np.load(model_file)

        assert len(weights_dict.files) == len(self.params_dict)

        for p_name, p in self.params_dict.iteritems():
            if p_name not in weights_dict:
                logging.error('parameter [%s] not in saved weights file', p_name)
            else:
                logging.info('loading parameter [%s]', p_name)
                p.set_value(weights_dict[p_name])
