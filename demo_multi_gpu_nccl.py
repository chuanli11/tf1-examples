import tensorflow as tf
import numpy as np

from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)


def get_shape(tensor):
    return tensor.get_shape().as_list()


# Source:
# https://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow
def get_available_gpus():
    """
        Returns a list of the identifiers of all visible GPUs.
    """
    from tensorflow.python.client import device_lib
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


PS_OPS = [
    'Variable', 'VariableV2', 'AutoReloadVariable', 'MutableHashTable',
    'MutableHashTableOfTensors', 'MutableDenseHashTable'
]


# see https://github.com/tensorflow/tensorflow/issues/9517
def assign_to_device(device, ps_device):
    """Returns a function to place variables on the ps_device.

    Args:
        device: Device for everything but variables
        ps_device: Device to put the variables on. Example values are /GPU:0 and /CPU:0.

    If ps_device is not set then the variables will be placed on the default device.
    The best device for shared varibles depends on the platform as well as the
    model. Start with CPU:0 and then test GPU:0 to see if there is an
    improvement.
    """
    def _assign(op):
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            return ps_device
        else:
            return device
    return _assign


def device_options():
    config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True)
    with tf.Session(config=config) as sess:
        # your code here
        pass


# Source:
# https://github.com/tensorflow/models/blob/master/tutorials/image/cifar10/cifar10_multi_gpu_train.py#L101
def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
    tower_grads: List of lists of (gradient, variable) tuples. The outer list ranges
        over the devices. The inner list ranges over the different variables.
    Returns:
            List of pairs of (gradient, variable) where the gradient has been averaged
            across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):

        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = [g for g, _ in grad_and_vars]
        grad = tf.reduce_mean(grads, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


# Source
# https://github.com/tensorpack/tensorpack/blob/81a4fc332eeae7e230e0736958014a0958fca822/tensorpack/graph_builder/utils.py#L140-L170
#################################
def split_grad_list(grad_list):
    """
    Args:
        grad_list: K x N x 2
    Returns:
        K x N: gradients
        K x N: variables
    """
    g = []
    v = []
    for tower in grad_list:
        g.append([x[0] for x in tower])
        v.append([x[1] for x in tower])
    print("g: ")
    print(g)
    print("v: ")
    print(v)
    return g, v


def merge_grad_list(all_grads, all_vars):
    """
    Args:
        all_grads (K x N): gradients
        all_vars(K x N): variables
    Return:
        K x N x 2: list of list of (grad, var) pairs
    """
    all_towers = [list(zip(gs, vs)) for gs, vs in zip(all_grads, all_vars)]
    print("all_towers: ")
    print(all_towers)
    return all_towers


def allreduce_grads(all_grads, average=True):
    """
    All-reduce average the gradients among K devices. Results are broadcasted to all devices.
    Args:
        all_grads (K x N): List of list of gradients. N is the number of variables.
        average (bool): average gradients or not.
    Returns:
        K x N: same as input, but each grad is replaced by the average over K devices.
    """
    from tensorflow.python.ops.nccl_ops import all_sum
    nr_tower = len(all_grads)
    if nr_tower == 1:
        return all_grads
    new_all_grads = []  # N x K
    for grads in zip(*all_grads):
        print('grads:')
        print(grads)
        summed = all_sum(grads)

        grads_for_devices = []  # K
        for g in summed:
            with tf.device(g.device):
                # tensorflow/benchmarks didn't average gradients
                if average:
                    g = tf.multiply(g, 1.0 / nr_tower)
            grads_for_devices.append(g)
        new_all_grads.append(grads_for_devices)

    # transpose to K x N
    ret = list(zip(*new_all_grads))
    return ret
###########################

# Source
# https://github.com/tensorpack/tensorpack/blob/81a4fc332eeae7e230e0736958014a0958fca822/tensorpack/graph_builder/training.py#L240
def apply_gradients(opt, grads):
    devices = get_available_gpus()
    # num_gpus = len(devices)
    raw_devices = ['{}'.format(k) for k in devices]
    # optimizer using NCCL
    train_ops = []
    with tf.name_scope('apply_gradients'):
        for idx, grad_and_vars in enumerate(grads):
            with tf.device(raw_devices[idx]):

                # apply_gradients may create variables. Make them LOCAL_VARIABLES
                # with override_to_local_variable(enable=idx > 0):
                # train_ops.append(opt.apply_gradients(
                #     grad_and_vars, name='apply_grad_{}'.format(idx)))

                # Check gradient overflow before update parameters
                with tf.name_scope('CheckOverflow'):
                    grad_ok = tf.reduce_all(tf.stack([tf.reduce_all(tf.is_finite(g)) for g, v in grad_and_vars]))
                train_ops.append(tf.cond(grad_ok, lambda:opt.apply_gradients(grad_and_vars, name='apply_grad_{}'.format(idx)), tf.no_op))
    train_op = tf.group(*train_ops, name='train_op')
    return train_op


def create_parallel_optimization(X, Y, input_flattened_dim, num_classes, optimizer, controller="/cpu:0"):
    devices = get_available_gpus()
    num_gpus = len(devices)
    # Place all ops on CPU by default
    # with tf.device('/cpu:0'):
    tower_grads = []
    losses = []
    # tf Graph input
    # X = tf.placeholder(tf.float32, [None, num_input])
    # Y = tf.placeholder(tf.float32, [None, num_classes])

    # Split data between GPUs
    X_s = tf.split(X, num_gpus)
    Y_s = tf.split(Y, num_gpus)
    # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    # Get the current variable scope so we can reuse all variables we need once we get
    # to the second iteration of the loop below
    with tf.variable_scope(tf.get_variable_scope()) as outer_scope:
        for i, id in enumerate(devices):
            name = 'tower_{}'.format(i)
            # Use the assign_to_device function to ensure that variables are created on the
            # controller.
            with tf.device(assign_to_device(id, controller)), tf.name_scope(name):

                _x = X_s[i]
                _y = Y_s[i]

                logits = mnist_mlp(_x, input_flattened_dim, num_classes)

                # Define loss and optimizer (with train logits, for dropout to take effect)
                loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
                    logits=logits, labels=_y))

                with tf.name_scope("compute_gradients"):
                    # `compute_gradients` returns a list of (gradient, variable) pairs
                    grads = optimizer.compute_gradients(loss)
                    tower_grads.append(grads)

                losses.append(loss)
                outer_scope.reuse_variables()

    print(tower_grads[0])

    all_grads, all_vars = split_grad_list(tower_grads)
    all_grads = allreduce_grads(all_grads)
    tower_grads = merge_grad_list(all_grads, all_vars)

    train_op = apply_gradients(optimizer, tower_grads)
    # return confusion_mat, equality_op, accuracy_op, train_op
    return train_op


def mnist_mlp(x, input_flattened_dim, num_classes):

    W = tf.get_variable(name='W1',
                        shape=[input_flattened_dim, num_classes],
                        initializer=tf.keras.initializers.glorot_uniform(seed=None),
                        dtype=tf.float32)
    b = tf.get_variable(name='b1',
                        shape=[num_classes],
                        initializer=tf.zeros_initializer(),
                        dtype=tf.float32)
    # Output layer, class prediction
    logits = tf.nn.relu(tf.matmul(x, W) + b)

    return logits

input_flattened_dim = 784
num_classes = 10
learning_rate = .0001
max_steps = 10000
batch_size = 200
X = tf.placeholder(tf.float32, [None, input_flattened_dim])
Y = tf.placeholder(tf.float32, [None, num_classes])


optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = create_parallel_optimization(X, Y, input_flattened_dim, num_classes, optimizer, controller="/cpu:0")


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(max_steps):
        print(i)
        # batch_x, batch_y = sess.run(next_element)
        batch_x, batch_y = mnist.train.next_batch(batch_size)
        # print(type(batch_x))
        # print(batch_x.shape)
        # print(type(batch_y))
        # print(batch_y.shape)
        # print('------------------------------------')
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
