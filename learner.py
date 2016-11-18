from nn.utils.config_factory import config
from nn.utils.generic_utils import *

import logging
import numpy as np
import sys
import time

import decoder
import evaluation
from dataset import *
from config import *


class Learner(object):
    def __init__(self, model, train_data, val_data=None):
        self.model = model
        self.train_data = train_data
        self.val_data = val_data

        logging.info('initial learner with training set [%s] (%d examples)',
                     train_data.name,
                     train_data.count)
        if val_data:
            logging.info('validation set [%s] (%d examples)', val_data.name, val_data.count)

    def train(self, shuffle=True):
        dataset = self.train_data
        nb_train_sample = dataset.count
        index_array = np.arange(nb_train_sample)

        nb_epoch = MAX_EPOCH
        batch_size = BATCH_SIZE

        logging.info('begin training')
        cum_updates = 0
        patience_counter = 0
        early_stop = False
        history_valid_perf = []
        best_model_params = None

        for epoch in range(nb_epoch):
            if shuffle:
                np.random.shuffle(index_array)

            batches = make_batches(nb_train_sample, batch_size)

            # epoch begin
            sys.stdout.write('Epoch %d' % epoch)
            begin_time = time.time()
            cum_nb_examples = 0
            loss = 0.0

            for batch_index, (batch_start, batch_end) in enumerate(batches):
                cum_updates += 1

                batch_ids = index_array[batch_start:batch_end]
                examples = dataset.get_examples(batch_ids)
                cur_batch_size = len(examples)

                inputs = dataset.get_prob_func_inputs(batch_ids)

                train_func_outputs = self.model.train_func(*inputs)
                batch_loss = train_func_outputs[0]
                logging.debug('prob_func finished computing')

                cum_nb_examples += cur_batch_size
                loss += batch_loss * batch_size

                logging.debug('Batch %d, avg. loss = %f', batch_index, batch_loss)

                if batch_index == 4:
                    elapsed = time.time() - begin_time
                    eta = nb_train_sample / (cum_nb_examples / elapsed)
                    print ', eta %ds' % (eta)
                    sys.stdout.flush()

                if cum_updates % VALID_PER_MINIBATCH == 0:
                    logging.info('begin validation')
                    decode_results = decoder.decode_dataset(self.model, self.val_data, verbose=False)
                    bleu, acc = evaluation.evaluate_decode_results(self.val_data, decode_results, verbose=False)

                    logging.info('sentence level bleu: %f', bleu)
                    logging.info('accuracy: %f', acc)

                    history_valid_perf.append(acc)
                    if acc >= np.array(history_valid_perf).max():
                        best_model_params = self.model.pull_params()
                        patience_counter = 0
                        logging.info('save current best model')
                        self.model.save('model.npz')
                    else:
                        patience_counter += 1
                        logging.info('hitting patience_counter: %d', patience_counter)
                        if patience_counter > TRAIN_PATIENCE:
                            logging.info('Early Stop!')
                            early_stop = True
                            break

                if cum_updates % SAVE_PER_MINIBATCH == 0:
                    self.model.save('model.iter%d' % cum_updates)

            logging.info('[Epoch %d] cumulative loss = %f, (took %ds)',
                         epoch,
                         loss / cum_nb_examples,
                         time.time() - begin_time)

            if early_stop:
                break

        logging.info('training finished, save the best model')
        np.savez('model.npz', **best_model_params)