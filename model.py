import tensorflow as tf
from tensorflow.contrib.layers import flatten

def SignModel(x, n_classes):
    # Hyperparameters
    mu = 0
    sigma = 0.1

    # Convolution Layer 1. Input = 32x32x1. Output = 28x28x108.
    conv1_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 1, 108), 
                        mean = mu, stddev = sigma), name='conv1_W')
    conv1_b = tf.Variable(tf.zeros(108), name='conv1_b')
    conv1   = tf.nn.conv2d(x, conv1_W, strides=[1, 1, 1, 1], 
                        padding='VALID') + conv1_b
    conv1 = tf.nn.tanh(conv1)
    
    # Pooling Layer 1. Input = 28x28x108. Output = 14x14x108.
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], 
                        strides=[1, 2, 2, 1], padding='VALID')
    
    # Local response normalization
    conv1 = tf.nn.local_response_normalization(conv1, name='LRN_1')

    # Convolution Layer 2. Input = 14x14x108 Output = 10x10x200.
    conv2_W = tf.Variable(tf.truncated_normal(shape=(5, 5, 108, 200), 
                        mean = mu, stddev = sigma), name='conv2_W')
    conv2_b = tf.Variable(tf.zeros(200), name='conv2_b')
    conv2   = tf.nn.conv2d(conv1, conv2_W, strides=[1, 1, 1, 1], 
                        padding='VALID') + conv2_b
    conv2 = tf.nn.tanh(conv2)

    # Pooling Layer 2. Input = 10x10x200. Output = 5x5x200.
    conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], 
                        strides=[1, 2, 2, 1], padding='VALID')

    # Local response normalization
    conv2 = tf.nn.local_response_normalization(conv2, name='LRN_2')
    
    # Pass both stage 1 and stage 2 features to classifier
    # Flatten and concatenate
    fc1 = tf.concat(1, [flatten(conv1), flatten(conv2)])
    fc1_shape = (fc1.get_shape().as_list()[-1], 100)

    # Fully Connected Layer 1. Output = 100.
    fc1_W     = tf.Variable(tf.truncated_normal(shape=(fc1_shape),
                            mean = mu, stddev = sigma), name='fc1_W')
    fc1_b     = tf.Variable(tf.zeros(100), name='fc1_b')
    fc1       = tf.matmul(fc1, fc1_W) + fc1_b
    fc1 = tf.nn.tanh(fc1)

    # Fully Connected Layer 2. Input = 100. Output = n_classes.
    fc2_W  = tf.Variable(tf.truncated_normal(shape=(100, n_classes),
                            mean = mu, stddev = sigma), name='fc2_W')
    fc2_b  = tf.Variable(tf.zeros(n_classes), name='fc2_b')
    logits = tf.matmul(fc1, fc2_W) + fc2_b

    return logits