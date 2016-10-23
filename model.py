import theano
import theano.tensor as T

from nn.layers.embeddings import Embedding
from nn.layers.core import Dense
from nn.layers.recurrent import BiLSTM, LSTM, CondAttLSTM
from nn.utils.theano_utils import ndim_itensor, tensor_right_shift

VOCAB_SIZE = 3000
SYMBOL_NUM = 100
RULE_NUM = 1000
EMBED_DIM = 300
RULE_EMBED_DIM = 200
WORD_EMBED_DIM = 100
QUERY_DIM = 200
LSTM_STATE_DIM = 300
DECODER_ATT_HIDDEN_DIM = 50


class Model:
    def __int__(self):
        self.symbol_embedding = Embedding(SYMBOL_NUM, EMBED_DIM, name='symbol_embed')

        self.query_embedding = Embedding(VOCAB_SIZE, EMBED_DIM, name='symbol_embed')

        self.rule_encoder_lstm = BiLSTM(EMBED_DIM, RULE_EMBED_DIM / 2, return_sequences=False,
                                        name='rule_encoder_lstm')

        self.query_encoder_lstm = LSTM(WORD_EMBED_DIM, QUERY_DIM, return_sequences=True,
                                       name='query_encoder_lstm')

        self.decoder_lstm = CondAttLSTM(RULE_EMBED_DIM, LSTM_STATE_DIM, QUERY_DIM, DECODER_ATT_HIDDEN_DIM,
                                        name='decoder_lstm')

        self.decoder_softmax = Dense(LSTM_STATE_DIM, RULE_NUM, activation='softmax', name='decoder_softmax')

        self.params = self.symbol_embedding.params + self.query_embedding.params + self.rule_encoder_lstm.params + \
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
        # (batch_size, max_example_rule_num, max_rule_length)
        rules = ndim_itensor(3, 'rules')

        # (batch_size, max_example_rule_num, max_rule_length, symbol_embed_dim)
        # (batch_size, max_example_rule_num, max_rule_length)
        rule_symbol_embed, rule_symbol_embed_mask = self.symbol_embedding(rules, mask_zero=True)

        # (batch_size, max_query_length)
        query_tokens = ndim_itensor(2, 'queries')

        # (batch_size, max_query_length, query_token_embed_dim)
        # (batch_size, max_query_length)
        query_token_embed, query_token_embed_mask = self.query_embedding(query_tokens, mask_zero=True)

        # (batch_size, max_example_rule_num, rule_embed_dim)
        rule_embed = self.get_rule_embedding(rule_symbol_embed, rule_symbol_embed_mask)

        rule_embed_shifted = tensor_right_shift(rule_embed)

        # (batch_size, max_query_length, query_embed_dim)
        query_embed = self.query_encoder_lstm(query_token_embed, mask=query_token_embed_mask)
        
        # (batch_size, max_example_rule_num, lstm_hidden_state)
        decoder_hidden_states, ctx_vectors = self.decoder_lstm(rule_embed_shifted, context=query_embed,
                                                               context_mask=query_token_embed_mask)

        # (batch_size, max_example_rule_num, rule_num)
        decoder_predict = self.decoder_softmax(decoder_hidden_states)

        batch_size = decoder_predict.shape[0]
        loss = decoder_predict[T.shape_padright(T.arange(batch_size)),
                               T.shape_padleft(T.arange(decoder_predict.shape[1])),
                               rules]

        loss = - (T.log(loss) * query_token_embed_mask).sum(axis=-1) / tgt_char_seq_mask.sum(axis=-1)

        char_gen_prob = T.mean(tgt_seq_prob)
