import numpy as np
import tensorflow as tf
import gzip
import pickle
import os
import sys
sys.path.extend(['alg/'])
import vcl
import coreset
import utils
from copy import deepcopy

tf.autograph.set_verbosity(0)
tf.logging.set_verbosity(tf.logging.ERROR)

class LongPermutedMnistGenerator():
    def __init__(self, max_iter=10, random_seed=0):
        # Open data file
        f = gzip.open('data/mnist.pkl.gz', 'rb')
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        f.close()

        # Define train and test data
        self.X_train = np.vstack((train_set[0], valid_set[0]))#[:1024]
        self.Y_train = np.hstack((train_set[1], valid_set[1]))#[:1024]
        self.X_test = test_set[0]#[:1024]
        self.Y_test = test_set[1]#[:1024]
        self.random_seed = random_seed
        self.max_iter = max_iter
        self.cur_iter = 0

        self.out_dim = 10           # Total number of unique classes
        self.class_list = list(range(10)) # List of unique classes being considered, in the order they appear

        # self.classes is the classes (with correct indices for training/testing) of interest at each task_id
        self.classes = []
        for iter in range(self.max_iter):
            self.classes.append(list(range(0,10)))

        self.sets = self.classes

        self.switch_points = [j for j in [10, 16, 20, 30, 35, 40] if j <= self.max_iter]  # at these indices the dataset will be permuted
        self.tasks_to_test = [0] + self.switch_points
        self.examples_per_iter = 1024
        # First task is (unpermuted) MNIST, subsequent tasks are random permutations of pixels
        self.perm_indices = [list(range(self.X_train.shape[1]))]
        for i, _ in enumerate(self.switch_points):
            np.random.seed(i+self.random_seed)
            perm_inds = list(range(self.X_train.shape[1]))
            np.random.shuffle(perm_inds)
            self.perm_indices.append(perm_inds)
        # make sure they are different permutations
        assert(len(set(tuple(perm_inds) for perm_inds in self.perm_indices)) == len(self.perm_indices))

        self.idx_map = {}
        self.batch_indices = []
        last_switch_point = 0
        for i, switch_point in enumerate((self.switch_points + [self.max_iter])):
            np.random.seed(i+self.random_seed)
            batch_inds = list(range(self.X_train.shape[0]))
            np.random.shuffle(batch_inds)
            for j in range(last_switch_point, switch_point):
                self.idx_map[j] = i
                self.batch_indices.append(batch_inds[(j-last_switch_point)*self.examples_per_iter:(j-last_switch_point+1)*self.examples_per_iter])
            last_switch_point = switch_point

    def get_dims(self):
        # Get data input and output dimensions
        return self.X_train.shape[1], self.out_dim

    def next_task(self):
        if self.cur_iter >= self.max_iter:
            raise Exception('Number of tasks exceeded!')
        else:
            #np.random.seed(self.cur_iter+self.random_seed)
            #perm_inds = list(range(self.X_train.shape[1]))
            # First task is (unpermuted) MNIST, subsequent tasks are random permutations of pixels
            #if self.cur_iter > 0:
            #    np.random.shuffle(perm_inds)
            perm_inds = self.perm_indices[self.idx_map[self.cur_iter]]
            batch_inds = self.batch_indices[self.cur_iter]

            # Retrieve train data
            #next_x_train = deepcopy(self.X_train)
            #next_x_train = next_x_train[batch_inds,:][:,perm_inds]
            next_x_train = deepcopy(self.X_train[batch_inds,:][:,perm_inds])

            # Initialise next_y_train to zeros, then change relevant entries to ones, and then stack
            next_y_train = np.zeros((len(next_x_train), 10))
            next_y_train[:,0:10] = np.eye(10)[self.Y_train[batch_inds]]

            # Retrieve test data
            if self.cur_iter in ([0] + self.switch_points):
                next_x_test = deepcopy(self.X_test)
                next_x_test = next_x_test[:,perm_inds]

                next_y_test = np.zeros((len(next_x_test), 10))
                next_y_test[:,0:10] = np.eye(10)[self.Y_test]
            else:
                next_x_test, next_y_test = None, None

            self.cur_iter += 1

            return next_x_train, next_y_train, next_x_test, next_y_test

    def reset(self):
        self.cur_iter = 0


store_weights = True        # Store weights after training on each task (for plotting later)
multi_head = False          # Multi-head or single-head network

hidden_size = [100, 100]    # Size and number of hidden layers
batch_size = 1024           # Batch size
no_epochs = 2500             # Number of training epochs per task
learning_rate = 0.005
permuted_num_tasks = 50 #10

options = [  # (diffusion, jump_bias, path_suffix, relative_broadening, beam_size)
    (1.2,    0.0, "long_m1.2", True, 2),         # 0
    (1.1,    0.0, "long_m1.1", True, 2),         # 1
    (1.05,    0.0, "long_m1.05", True, 2),       # 2
    (1.01,    0.0, "long_m1.01", True, 2),       # 3
    (1.005,    0.0, "long_m1.005", True, 2),     # 4
    (1.001,    0.0, "long_m1.001", True, 2),     # 5
    (1.0001,   0.0, "long_m1.0001", True, 2),    # 6
    (0.0,   0.0, "baseline", False, 1),          # 7
    (1.5,    0.0, "long_m1.5", True, 2),         # 8
    (2.0,    0.0, "long_m2.0", True, 2),         # 9
]

import sys
diffusion = float(sys.argv[-1])#, int(sys.argv[-1])]
beam_size = 2 #1
path_suffix = f"long_m{diffusion}"
jump_bias = 0.2 #0.0
mult_diff = diffusion != 0.0

# No coreset
tf.reset_default_graph()
random_seed = 1
tf.set_random_seed(random_seed+1)
np.random.seed(random_seed)

path = f'model_storage/long_permuted/{path_suffix}/'    # Path where to store files
#path = f'model_storage/long_small_permuted/{path_suffix}/'    # Path where to store files
beam_path = f'{path}beam_j{jump_bias}_history.pkl'
# Ensure path exists to save results to
inc_path = "."
for folder in path.split("/"):
    inc_path += "/"
    inc_path += folder
    if not os.path.exists(inc_path):
        print(f"Making directory: {inc_path}")
        os.mkdir(inc_path)

data_gen = LongPermutedMnistGenerator(max_iter=permuted_num_tasks, random_seed=random_seed)
coreset_size = 0
vcl_result = vcl.run_vcl_shared(hidden_size, no_epochs, data_gen,
    coreset.rand_from_batch, coreset_size, batch_size, path, multi_head, store_weights=store_weights,
    beam_size=beam_size, diffusion=diffusion, jump_bias=jump_bias, mult_diff=mult_diff, beam_path=beam_path)

# Store accuracies
np.savez(path + 'test_acc.npz', acc=vcl_result)


# # Random coreset
# tf.reset_default_graph()
# random_seed = 1
# tf.set_random_seed(random_seed+1)
# np.random.seed(random_seed)

# path = 'model_storage/permuted_coreset/'    # Path where to store files
# data_gen = PermutedMnistGenerator(max_iter=permuted_num_tasks, random_seed=random_seed)
# coreset_size = 200
# vcl_result_coresets = vcl.run_vcl_shared(hidden_size, no_epochs, data_gen,
#     coreset.rand_from_batch, coreset_size, batch_size, path, multi_head, store_weights=store_weights)

# # Store accuracies
# np.savez(path + 'test_acc.npz', acc=vcl_result_coresets)

# # Plot average accuracy
# utils.plot('model_storage/permuted_mnist_', vcl_result, vcl_result_coresets)