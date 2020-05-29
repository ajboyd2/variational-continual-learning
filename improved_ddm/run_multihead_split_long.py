import numpy as np
import tensorflow as tf
import gzip
import pickle
import sys
sys.path.extend(['alg/'])
import vcl
import coreset
import utils
import os


class OnlineMnistDataSimulator():
    def __init__(self, max_iter=50, random_seed=0):
        f = gzip.open('data/mnist.pkl.gz', 'rb')
        train_set, valid_set, test_set = pickle.load(f, encoding='latin1')
        f.close()

        self.X_train = np.vstack((train_set[0], valid_set[0]))
        self.X_test = test_set[0]
        self.max_iter = max_iter
        
        self.out_dim = 10           # Total number of unique classes
        self.class_list = list(range(10)) # List of unique classes being considered, in the order they appear

        # self.classes is the classes (with correct indices for training/testing) of interest at each task_id
        self.classes = []
        for iter in range(self.max_iter):
            self.classes.append(list(range(0,10)))

        self.sets = self.classes

        self.train_label = np.hstack((train_set[1], valid_set[1]))
        self.test_label = test_set[1]
        
        self._sets_0 = [0, 2, 4, 6, 8, 0, 1]
        self._sets_1 = [1, 3, 5, 7, 9, 2, 3]
        
        self.max_iter = max_iter #len(self.sets_0)
        self.switch_points = [j for j in [10, 16, 20, 30, 35, 40] if j <= self.max_iter]
        self.tasks_to_test = [0] + self.switch_points
        
        assert len(self.switch_points) == len(self._sets_0) - 1
        self.sets_0 = []
        self.sets_1 = []
        
        last_switch_point = 0
        for i, switch_point in enumerate((self.switch_points + [self.max_iter])):
            for j in range(last_switch_point, switch_point):
                self.sets_0.append(self._sets_0[i])
                self.sets_1.append(self._sets_1[i])
            last_switch_point = switch_point
        assert len(self.sets_0) == self.max_iter
        
        self.examples_per_iter = 1024
        self.random_seed = random_seed
        self.rng = np.random.RandomState(self.random_seed)
        self.cur_train_iter = 0
        self.cur_test_iter = 0
    
    def get_dims(self):
        # Get data input and output dimensions
        return self.X_train.shape[1], self.out_dim

    def _convert_to_full_output(self, y_train, cur_iter):
        new_y_train = np.zeros((y_train.shape[0], 10))
        new_y_train[:,self.sets_0[cur_iter]] = y_train[:,0]
        new_y_train[:,self.sets_1[cur_iter]] = y_train[:,1]
        return new_y_train

    def next_task(self):
        train_batch, test_batch = self.next_train_batch(True), self.next_test_batch(True)
        return train_batch[0], train_batch[1], test_batch[0], test_batch[1]
    
    def next_train_batch(self, full_output=False):
        if self.cur_train_iter >= self.max_iter:
            raise Exception('No more training data to return!')
        else:
            train_0_id = self.rng.choice(np.where(self.train_label == self.sets_0[self.cur_train_iter])[0], size=self.examples_per_iter, replace=False)
            train_1_id = self.rng.choice(np.where(self.train_label == self.sets_1[self.cur_train_iter])[0], size=self.examples_per_iter, replace=False)
    
            next_x_train = np.vstack((self.X_train[train_0_id], self.X_train[train_1_id]))
            next_y_train = np.vstack((np.ones((train_0_id.shape[0], 1)), np.zeros((train_1_id.shape[0], 1))))
            next_y_train = np.hstack((next_y_train, 1-next_y_train))
    
            if full_output:
                next_y_train = self._convert_to_full_output(next_y_train, self.cur_train_iter)
    
            self.cur_train_iter += 1
            return next_x_train, next_y_train
    
    def reset_train(self):
        self.cur_train_iter = 0
    
    def reset_test(self):
        self.cur_test_iter = 0
    
    def next_test_batch(self, full_output=False):
        if self.cur_test_iter >= self.max_iter:
            raise Exception('No more test data to return!')
        else:       
            # Retrieve test data
            if self.cur_test_iter in ([0] + self.switch_points):
                test_0_id = np.where(self.test_label == self.sets_0[self.cur_test_iter])[0]
                test_1_id = np.where(self.test_label == self.sets_1[self.cur_test_iter])[0]
        
                next_x_test = np.vstack((self.X_test[test_0_id], self.X_test[test_1_id]))
                next_y_test = np.vstack((np.ones((test_0_id.shape[0], 1)), np.zeros((test_1_id.shape[0], 1))))
                next_y_test = np.hstack((next_y_test, 1-next_y_test))

                if full_output:
                    next_y_test = self._convert_to_full_output(next_y_test, self.cur_test_iter)
            else:
                next_x_test, next_y_test = None, None

            self.cur_test_iter += 1
            return next_x_test, next_y_test



store_weights = True    # Store weights after training on each task (for plotting later)
multi_head = False #True       # Multi-head or single-head network

hidden_size = [200]     # Size and number of hidden layers
batch_size = 256        # Batch size
no_epochs = 1200         # Number of training epochs per task
learning_rate = 0.005
permuted_num_tasks = 50 #10


# No coreset
tf.reset_default_graph()
random_seed = 0
tf.set_random_seed(random_seed+1)
np.random.seed(random_seed)


import sys
diffusion = float(sys.argv[-2])#, int(sys.argv[-1])]
beam_size = int(sys.argv[-1]) #1
jump_bias = 0.2 #0.0
mult_diff = diffusion != 0.0
upper_fixed = False 

# No coreset
tf.reset_default_graph()
random_seed = 1
tf.set_random_seed(random_seed+1)
np.random.seed(random_seed)

restart_every_iter = False
restart_every_switch = False

if restart_every_switch:
    path_suffix = f"long_m{diffusion}_repeat_switch"
elif restart_every_iter:
    path_suffix = f"long_m{diffusion}_repeat_iter"
else:
    path_suffix = f"long_m{diffusion}"

path = f'model_storage/long_split{"_upper_fixed" if upper_fixed else ""}/{path_suffix}/'    # Path where to store files
#path = f'model_storage/long_small_permuted/{path_suffix}/'    # Path where to store files
beam_path = f'{path}beam_s{beam_size}_j{jump_bias}_history.pkl'
# Ensure path exists to save results to
inc_path = "."
for folder in path.split("/"):
    inc_path += "/"
    inc_path += folder
    if not os.path.exists(inc_path):
        print(f"Making directory: {inc_path}")
        os.mkdir(inc_path)

print(f"Saving results to {beam_path}.")

data_gen = OnlineMnistDataSimulator(max_iter=permuted_num_tasks, random_seed=random_seed)
coreset_size = 0
vcl_result = vcl.run_vcl_shared(hidden_size, no_epochs, data_gen,
    coreset.rand_from_batch, coreset_size, batch_size, path, multi_head, store_weights=store_weights,
    beam_size=beam_size, diffusion=diffusion, jump_bias=jump_bias, mult_diff=mult_diff, beam_path=beam_path,
    upper_fixed=upper_fixed, restart_every_iter=restart_every_iter, restart_every_switch=restart_every_switch)

# Store accuracies
np.savez(path + 'test_acc.npz', acc=vcl_result)








path = 'model_storage/split/'   # Path where to store files
data_gen = SplitMnistGenerator()
coreset_size = 0
vcl_result = vcl.run_vcl_shared(hidden_size, no_epochs, data_gen,
    coreset.rand_from_batch, coreset_size, batch_size, path, multi_head, store_weights=store_weights)

# Store accuracies
np.savez(path + 'test_acc.npz', acc=vcl_result)


# Random coreset
# tf.reset_default_graph()
# random_seed = 0
# tf.set_random_seed(random_seed+1)
# np.random.seed(random_seed)

# path = 'model_storage/split_coreset/'   # Path where to store files
# data_gen = SplitMnistGenerator()
# coreset_size = 40
# vcl_result_coresets = vcl.run_vcl_shared(hidden_size, no_epochs, data_gen,
#     coreset.rand_from_batch, coreset_size, batch_size, path, multi_head, store_weights=store_weights)

# # Store accuracies
# np.savez(path + 'test_acc.npz', acc=vcl_result_coresets)

# # Plot average accuracy
# utils.plot('model_storage/split_mnist_', vcl_result, vcl_result_coresets)