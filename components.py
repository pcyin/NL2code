import theano
import theano.tensor as T
import numpy as np

from nn.layers.embeddings import Embedding
from nn.layers.core import Dense, Layer
from nn.layers.recurrent import BiLSTM, LSTM, CondAttLSTM
from nn.utils.theano_utils import ndim_itensor, tensor_right_shift, ndim_tensor
import nn.optimizers as optimizers

from config import *
from grammar import *
from parse import *
from tree import *


class PointerNet(Layer):
    def __init__(self, name='PointerNet'):
        super(PointerNet, self).__init__()

        self.dense1_input = Dense(QUERY_DIM, POINTER_NET_HIDDEN_DIM, activation='linear', name='Dense1_input')

        self.dense1_h = Dense(LSTM_STATE_DIM, POINTER_NET_HIDDEN_DIM, activation='linear', name='Dense1_h')

        self.dense2 = Dense(POINTER_NET_HIDDEN_DIM, 1, activation='linear', name='Dense2')

        self.params += self.dense1_input.params + self.dense1_h.params + self.dense2.params

        self.set_name(name)

    def __call__(self, query_embed, query_token_embed_mask, decoder_hidden_states):
        query_embed_trans = self.dense1_input(query_embed)
        h_trans = self.dense1_h(decoder_hidden_states)

        query_embed_trans = query_embed_trans.dimshuffle((0, 'x', 1, 2))
        h_trans = h_trans.dimshuffle((0, 1, 'x', 2))

        # (batch_size, max_decode_step, query_token_num, ptr_net_hidden_dim)
        dense1_trans = T.tanh(query_embed_trans + h_trans)

        scores = self.dense2(dense1_trans).flatten(3)

        scores = T.exp(scores - T.max(scores, axis=-1, keepdims=True))
        scores *= query_token_embed_mask.dimshuffle((0, 'x', 1))
        scores = scores / T.sum(scores, axis=-1, keepdims=True)

        return scores

class Hyp:
    def __init__(self, tree=None):
        if not tree:
            self.tree = Tree('root')
        else:
            self.tree = tree

        self.score = 0.0

    def __repr__(self):
        return self.tree.__repr__()

    @staticmethod
    def can_expand(node):
        if node.holds_value and \
                (node.label and node.label.endswith('<eos>')):
            return False
        elif node.type == 'epsilon':
            return False
        elif is_terminal_ast_type(node.type):
            return False

        # if node.type == 'root':
        #     return True
        # elif inspect.isclass(node.type) and issubclass(node.type, ast.AST) and not is_terminal_ast_type(node.type):
        #     return True
        # elif node.holds_value and not node.label.endswith('<eos>'):
        #     return True

        return True

    def frontier_nt_helper(self, node):
        if node.is_leaf:
            if Hyp.can_expand(node):
                return node
            else:
                return None

        for child in node.children:
            result = self.frontier_nt_helper(child)
            if result:
                return result

        return None

    def frontier_nt(self):
        return self.frontier_nt_helper(self.tree)
