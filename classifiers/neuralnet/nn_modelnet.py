import nn_data
import tensorflow as tf
import time
from datetime import timedelta
import math
import random
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

beta=0.01

rate=0.3
batch_size = 32
path='ModelNet10pcd'
classes = os.listdir(path)
learn_rate=1e-4
# 20% of the data will automatically be used for validation

data = nn_data.read_train_sets()
shapes=nn_data.shapes()[2]
print shapes
session = tf.Session()
x = tf.placeholder(tf.float32, shape=[None, 1,shapes], name='x')

## labels
y_true = tf.placeholder(tf.float32, shape=[None,len(classes)], name='y_true')
y_true_cls = tf.argmax(y_true, axis=1)



##Network graph params


def create_weights(shape):
    return tf.Variable(tf.truncated_normal(shape,stddev=0.01))

def create_biases(size):
    return tf.Variable(tf.constant(0.01,shape=[size]))





    

def create_flatten_layer(layer):
    #We know that the shape of the layer will be [batch_size img_size img_size num_channels] 
    # But let's get it from the previous layer.
    layer_shape = layer.get_shape()

    ## Number of features will be img_height * img_width* num_channels. But we shall calculate it in place of hard-coding it.
    num_features = layer_shape[1:3].num_elements()

    ## Now, we Flatten the layer so we shall have to reshape to num_features
    layer = tf.reshape(layer, [-1, num_features])

    return layer


def create_fc_layer(input,          
             num_inputs,    
             num_outputs,
             use_relu=True):
    
    #Let's define trainable weights and biases.
    weights = create_weights(shape=[num_inputs, num_outputs])
    regularizers = tf.nn.l2_loss(weights)
    biases = create_biases(num_outputs)

    # Fully connected layer takes input x and produces wx+b.Since, these are matrices, we use matmul function in Tensorflow
    layer = tf.matmul(input, weights) + biases
    if use_relu:
        layer = tf.nn.relu(layer)

    return layer,regularizers



          
layer_flat = create_flatten_layer(x)
# dense1 = tf.layers.dense(inputs=layer_flat, units=2048, activation=tf.nn.relu)
# dropout = tf.layers.dropout(inputs=dense1, rate=0.4, training=True)
# # dense2=tf.layers.dense(inputs=dropout, units=2048,activation=tf.nn.relu)
# # dropout1=tf.layers.dropout(inputs=dense2, rate=0.4, training=True)
# # dense3=tf.layers.dense(inputs=dropout1,units=2048,activation=tf.nn.relu)
# # dropout2=tf.layers.dropout(inputs=dense3, rate=0.4, training=True)
# dense4=tf.layers.dense(inputs=dropout,units=len(classes))


num_layer=2048
layer_fc1,l1=create_fc_layer(input=layer_flat,
                     num_inputs=layer_flat.get_shape()[1:3].num_elements(),
                     num_outputs=num_layer,
                     use_relu=True)
dropout=tf.layers.dropout(inputs=layer_fc1,rate=rate,training=True)

layer_fc2,l2= create_fc_layer(input=dropout,
                     num_inputs=num_layer,
                     num_outputs=num_layer,
                     use_relu=True)
dropout1=tf.layers.dropout(inputs=layer_fc2,rate=rate,training=True)
layer_fc3,l3= create_fc_layer(input=dropout1,
                     num_inputs=num_layer,
                     num_outputs=num_layer,
                     use_relu=True)
dropout2=tf.layers.dropout(inputs=layer_fc3,rate=rate,training=True)
layer_fc4,l4= create_fc_layer(input=dropout2,
                     num_inputs=num_layer,
                     num_outputs=len(classes),
                     use_relu=False)
y_pred = tf.nn.softmax(layer_fc4,name='y_pred')
y_pred_cls = tf.argmax(y_pred, axis=1)
session.run(tf.global_variables_initializer())
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc4,
                                                    labels=y_true)

#### L2 Regularisation
# r=l1+l2+l3+l4
# cost = tf.reduce_mean(cross_entropy + beta*r)
#######

cost = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learn_rate).minimize(cost)
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


session.run(tf.global_variables_initializer()) 


def show_progress(epoch, feed_dict_train, feed_dict_validate, val_loss):
    acc = session.run(accuracy, feed_dict=feed_dict_train)
    val_acc = session.run(accuracy, feed_dict=feed_dict_validate)
    msg = "Training Epoch {0} --- Training Accuracy: {1:>6.1%}, Validation Accuracy: {2:>6.1%},  Validation Loss: {3:.3f}"
    print(msg.format(epoch + 1, acc, val_acc, val_loss))

total_iterations = 0

saver = tf.train.Saver()
def train(num_iteration):
    global total_iterations
    
    for i in range(total_iterations,
                   total_iterations + num_iteration):

        x_batch, y_true_batch = data.train.next_batch(batch_size)
        # print x_batch, y_true_batch, _, cls_batch
        x_valid_batch, y_valid_batch= data.valid.next_batch(batch_size)
        # print x_valid_batch, y_valid_batch, _, valid_cls_batch
        # print x_batch
        feed_dict_tr = {x: x_batch,
                           y_true: y_true_batch}
        feed_dict_val = {x: x_valid_batch,
                              y_true: y_valid_batch}

        session.run(optimizer, feed_dict=feed_dict_tr)

        if i % int(data.train.num_examples/batch_size) == 0: 
            val_loss = session.run(cost, feed_dict=feed_dict_val)
            epoch = int(i / int(data.train.num_examples/batch_size))    
            show_progress(epoch, feed_dict_tr, feed_dict_val, val_loss)
            saver.save(session, './nn_modelnet10') 


    total_iterations += num_iteration

train(num_iteration=2000)