import theano
import theano.tensor as T
import numpy as np

from collections import OrderedDict
import logging
import copy

from nn.layers.embeddings import Embedding
from nn.layers.core import Dense
from nn.layers.recurrent import BiLSTM, LSTM, CondAttLSTM
from nn.utils.theano_utils import ndim_itensor, tensor_right_shift, ndim_tensor
import nn.optimizers as optimizers

from config import *


class Model:
    def __init__(self):
        self.symbol_embedding = Embedding(RULE_NUM, EMBED_DIM, name='symbol_embed')

        self.query_embedding = Embedding(VOCAB_SIZE, EMBED_DIM, name='query_embed')

        self.rule_encoder_lstm = BiLSTM(EMBED_DIM, RULE_EMBED_DIM / 2, return_sequences=False,
                                        name='rule_encoder_lstm')

        self.query_encoder_lstm = LSTM(EMBED_DIM, QUERY_DIM, return_sequences=True,
                                       name='query_encoder_lstm')

        self.decoder_lstm = CondAttLSTM(RULE_EMBED_DIM, LSTM_STATE_DIM, QUERY_DIM, DECODER_ATT_HIDDEN_DIM,
                                        name='decoder_lstm')

        self.decoder_softmax = Dense(LSTM_STATE_DIM, RULE_NUM, activation='softmax', name='decoder_softmax')

        # self.rule_encoder_lstm.params
        self.params = self.symbol_embedding.params + self.query_embedding.params + \
                      self.query_encoder_lstm.params + self.decoder_lstm.params + self.decoder_softmax.params
        
    def get_rule_embedding(self, rule_embed, rule_embed_mask):
        batch_size = rule_embed.shape[0]
        max_example_rule_num = rule_embed.shape[1]

        # (batch_size, max_example_rule_num, symbol_embed_dim)
        rule_src = rule_embed[:, :, 0, :]

        # (batch_size, max_example_rule_num, max_rule_length - 1, symbol_embed_dim)
        rule_tgt = rule_embed[:, :, 1:, :]
        rule_embed_mask = rule_embed_mask[:, :, 1:]

        stack_size = rule_tgt.shape[0] * max_example_rule_num

        # (batch_size * max_example_rule_num, symbol_embed_dim)
        rule_src = rule_src.reshape((stack_size, -1))
        # (batch_size * max_example_rule_num, max_rule_length - 1, symbol_embed_dim)
        rule_tgt_stacked = rule_tgt.reshape((stack_size, rule_tgt.shape[2], -1))
        # (batch_size * max_example_rule_num, max_rule_length - 1)
        rule_embed_mask_stacked = rule_embed_mask.reshape((stack_size, rule_embed_mask[2]))

        # (batch_size * max_example_rule_num, rule_embed_dim)
        rule_symbol_embed = self.rule_encoder_lstm(rule_tgt_stacked, mask=rule_embed_mask_stacked, init_state=rule_src)
        # (batch_size, max_example_rule_num, rule_embed_dim)
        rule_symbol_embed = rule_symbol_embed.reshape((batch_size, max_example_rule_num, rule_symbol_embed.shape[1]))

        return rule_symbol_embed

    def build(self):
        # (batch_size, max_example_rule_num)
        rules = ndim_itensor(2, 'rules')

        # (batch_size, max_example_rule_num, rule_embed_dim)
        # (batch_size, max_example_rule_num)
        rule_embed, rule_embed_mask = self.symbol_embedding(rules, mask_zero=True)

        # (batch_size, max_query_length)
        query_tokens = ndim_itensor(2, 'queries')

        # (batch_size, max_query_length, query_token_embed_dim)
        # (batch_size, max_query_length)
        query_token_embed, query_token_embed_mask = self.query_embedding(query_tokens, mask_zero=True)

        rule_embed_shifted = tensor_right_shift(rule_embed)

        # (batch_size, max_query_length, query_embed_dim)
        query_embed = self.query_encoder_lstm(query_token_embed, mask=query_token_embed_mask)
        
        # (batch_size, max_example_rule_num, lstm_hidden_state)
        decoder_hidden_states, _, ctx_vectors = self.decoder_lstm(rule_embed_shifted, context=query_embed,
                                                                  context_mask=query_token_embed_mask)

        # (batch_size, max_example_rule_num, rule_num)
        decoder_predict = self.decoder_softmax(decoder_hidden_states)

        batch_size = decoder_predict.shape[0]
        loss = decoder_predict[T.shape_padright(T.arange(batch_size)),
                               T.shape_padleft(T.arange(decoder_predict.shape[1])),
                               rules]

        loss = - (T.log(loss) * rule_embed_mask).sum(axis=-1) / rule_embed_mask.sum(axis=-1)
        loss = T.mean(loss)

        # let's build the function!
        train_inputs = [query_tokens, rules]
        optimizer = optimizers.get('adam')
        updates, grads = optimizer.get_updates(self.params, loss)
        self.train_func = theano.function(train_inputs, loss, updates=updates)

        self.build_decoder(query_tokens, query_embed, query_token_embed_mask)

    def build_decoder(self, query_tokens, query_embed, query_token_embed_mask):
        # (batch_size, decoder_state_dim)
        decoder_prev_state = ndim_tensor(2, name='decoder_prev_state')

        # (batch_size, decoder_state_dim)
        decoder_prev_cell = ndim_tensor(2, name='decoder_prev_cell')

        # (batch_size)
        prev_rule = T.ivector(name='prev_y')

        prev_rule_embedding = T.switch(prev_rule[:, None] < 0,
                                       T.alloc(0., 1, EMBED_DIM),
                                       self.symbol_embedding(prev_rule))

        prev_rule_embedding = prev_rule_embedding.dimshuffle((0, 'x', 1))

        # (batch_size, 1, decoder_state_dim)
        # (batch_size, 1, decoder_state_dim)
        # (batch_size, 1, field_token_encode_dim)
        decoder_next_state, decoder_next_cell, ctx_vectors = self.decoder_lstm(prev_rule_embedding,
                                                                               init_state=decoder_prev_state,
                                                                               init_cell=decoder_prev_cell,
                                                                               context=query_embed,
                                                                               context_mask=query_token_embed_mask)

        decoder_next_state = decoder_next_state.flatten(2)
        decoder_next_cell = decoder_next_cell.flatten(2)
        decoder_predict = self.decoder_softmax(decoder_next_state)

        inputs = [query_tokens]
        outputs = [query_embed, query_token_embed_mask]

        self.decoder_func_init = theano.function(inputs, outputs)

        inputs = [decoder_prev_state, decoder_prev_cell, prev_rule,
                  query_embed, query_token_embed_mask]

        outputs = [decoder_next_state, decoder_next_cell, decoder_predict]

        self.decoder_func_next_step = theano.function(inputs, outputs)

    def decode(self, example, beam_size=30, max_time_step=40):
        # beam search decoding

        EOS = 1

        query_tokens, gold_rules = example.data

        query_embed, query_token_embed_mask = self.decoder_func_init(query_tokens)

        completed_hyps = []
        completed_hyp_scores = []
        completed_hyp_num = 0
        live_hyp_num = 1

        hyp_samples = [list() for i in range(live_hyp_num)]
        hyp_scores = np.zeros(live_hyp_num).astype('float32')

        decoder_prev_state = np.zeros((1, LSTM_STATE_DIM)).astype('float32')
        decoder_prev_cell = np.zeros((1, LSTM_STATE_DIM)).astype('float32')
        prev_rule = np.asarray([-1]).astype('int32')

        for t in range(max_time_step):
            # print 'time step [%d]' % t
            query_embed_tiled = np.tile(query_embed, [live_hyp_num, 1, 1])
            query_token_embed_mask_tiled = np.tile(query_token_embed_mask, [live_hyp_num, 1])

            inputs = [decoder_prev_state, decoder_prev_cell, prev_rule,
                      query_embed_tiled, query_token_embed_mask_tiled]

            decoder_next_state, decoder_next_cell, decoder_predict = self.decoder_func_next_step(*inputs)

            # (batch_size, max_rule_num)
            cand_scores = hyp_scores[:, None] - np.log(decoder_predict)
            cand_flat = cand_scores.flatten()
            ranks_flat = cand_flat.argsort()[:(beam_size - completed_hyp_num)]

            max_rule_num = decoder_predict.shape[1]
            hyp_indices = ranks_flat / max_rule_num
            rule_indices = ranks_flat % max_rule_num
            costs = cand_flat[ranks_flat]

            new_hyp_samples = []
            new_hyp_scores = np.zeros(beam_size - completed_hyp_num).astype('float32')
            new_hyp_states = []

            for idx, [ti, wi] in enumerate(zip(hyp_indices, rule_indices)):
                new_hyp_samples.append(hyp_samples[ti]+[wi])
                new_hyp_scores[idx] = copy.copy(costs[idx])
                new_hyp_states.append((copy.copy(decoder_next_state[ti]), copy.copy(decoder_next_cell[ti])))

            # check the finished samples
            new_live_hyp_num = 0
            hyp_samples = []
            hyp_scores = []
            hyp_states = []

            for idx in xrange(len(new_hyp_samples)):
                if new_hyp_samples[idx][-1] == EOS:
                    completed_hyps.append(new_hyp_samples[idx][:-1])  # remove ending EOS
                    completed_hyp_scores.append(new_hyp_scores[idx])
                    completed_hyp_num += 1
                else:
                    new_live_hyp_num += 1
                    hyp_samples.append(new_hyp_samples[idx])
                    hyp_scores.append(new_hyp_scores[idx])
                    hyp_states.append(new_hyp_states[idx])

            hyp_scores = np.array(hyp_scores)
            live_hyp_num = new_live_hyp_num

            if new_live_hyp_num < 1:
                break
            if completed_hyp_num >= beam_size:
                break

            prev_rule = np.array([w[-1] for w in hyp_samples]).astype('int32')
            decoder_prev_state = np.array([state[0] for state in hyp_states]).astype('float32')
            decoder_prev_cell = np.array([state[1] for state in hyp_states]).astype('float32')

        sorted_hyps = np.argsort(completed_hyp_scores)
        completed_hyps = [completed_hyps[i] for i in sorted_hyps]
        completed_hyp_scores = [completed_hyp_scores[i] for i in sorted_hyps]

        return completed_hyps, completed_hyp_scores

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

        for p_name, p in self.params_dict.iteritems():
            if p_name not in weights_dict:
                logging.error('parameter [%s] not in saved weights file', p_name)
            else:
                logging.info('loading parameter [%s]', p_name)
                p.set_value(weights_dict[p_name])
