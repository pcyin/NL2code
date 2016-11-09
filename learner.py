from nn.utils.config_factory import config
from nn.utils.generic_utils import *

import logging
import numpy as np
import sys
import time

from dataset import *
from config import *


class Learner(object):
    def __init__(self, model, train_data, test_data=None, val_data=None):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.val_data = val_data

        logging.info('initial learner with training data [%s] (%d examples)',
                     train_data.name,
                     train_data.count)

    def train(self, shuffle=True):
        dataset = self.train_data
        nb_train_sample = dataset.count
        index_array = np.arange(nb_train_sample)

        nb_epoch = EPOCH_NUM
        batch_size = BATCH_SIZE

        logging.info('begin training, # training examples: %d', nb_train_sample)

        for epoch in range(nb_epoch):
            if shuffle:
                np.random.shuffle(index_array)

            batches = make_batches(nb_train_sample, batch_size)

            # epoch begin
            sys.stdout.write('Epoch %d \n' % epoch)
            begin_time = time.time()
            cum_nb_examples = 0
            loss = 0.0

            for batch_index, (batch_start, batch_end) in enumerate(batches):
                batch_ids = index_array[batch_start:batch_end]
                examples = dataset.get_examples(batch_ids)
                cur_batch_size = len(examples)

                inputs = dataset.get_prob_func_inputs(batch_ids)

                train_func_outputs = self.model.train_func(*inputs)
                batch_loss = train_func_outputs[0]
                logging.debug('prob_func finished computing')

                cum_nb_examples += cur_batch_size
                loss += batch_loss * batch_size

                logging.info('Batch %d, avg. loss = %f', batch_index, batch_loss)

            logging.info('[Epoch %d] cumulative loss = %f, (took %ds)',
                         epoch,
                         loss / cum_nb_examples,
                         time.time() - begin_time)

            self.model.save('model.epoch%d' % epoch)