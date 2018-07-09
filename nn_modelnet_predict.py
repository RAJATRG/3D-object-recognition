import tensorflow as tf
import numpy as np
import os,glob
import sys,argparse
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.utils import shuffle
# pca=PCA(n_components=900,whiten=True)
# ###########################
path='ModelNet10pcd'
classes=os.listdir(path)
print classes
num_class=len(classes)
acc=0
true_label=[]
true_data=[]

for file in classes:
	train_path='test_fisher/'+file+'_fisher_test.txt'
	data=np.loadtxt(train_path,delimiter=',')
	shape=data.shape
	# data=data[:]
	# data=StandardScaler().fit_transform(data)
	label=np.zeros(num_class)
	index=classes.index(file)

	label[index]=1.0
	for i in range(data.shape[0]):
		true_data.append(data[i])
		true_label.append(label)

# true_data=np.array(true_data)
# true_data=shuffle(true_data)

# true_data=np.reshape(true_data,[data.shape[0],shape[1]])
# true_data,true_label=shuffle(true_data,true_label)
true_data=StandardScaler().fit_transform(true_data)
# pca.fit_transform(true_data)
# true_data=normalize(true_data, norm='l2')
true_data=np.array(true_data)
true_label=np.array(true_label)
true_data=true_data[:,np.newaxis,:]
x_batch=true_data

## Let us restore the saved model 
sess = tf.Session()
# Step-1: Recreate the network graph. At this step only graph is created.
saver = tf.train.import_meta_graph('nn_modelnet10.meta')
# Step-2: Now let's load the weights saved using the restore method.
saver.restore(sess, tf.train.latest_checkpoint('./'))

# Accessing the default graph which we have restored
graph = tf.get_default_graph()

# Now, let's get hold of the op that we can be processed to get the output.
# In the original network y_pred is the tensor that is the prediction of the network
y_pred = graph.get_tensor_by_name("y_pred:0")

## Let's feed the images to the input placeholders
x= graph.get_tensor_by_name("x:0") 
y_true = graph.get_tensor_by_name("y_true:0")
shape=true_label.shape 
y= np.zeros(shape) 


### Creating the feed_dict that is required to be fed to calculate y_pred 
feed_dict_testing = {x: x_batch, y_true: y}
result=sess.run(y_pred, feed_dict=feed_dict_testing)
# result is of this format [probabiliy_of_rose probability_of_sunflower]
b = np.zeros_like(result)
b[np.arange(len(result)), result.argmax(1)] = 1
# b=np.round(result)
accuracy_=accuracy_score(true_label,b)
# print file,accuracy_*100
acc=acc+accuracy_

print 'total accuracy',acc*100, '%'