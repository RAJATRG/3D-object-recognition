import os
import glob
from sklearn.utils import shuffle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
# pca=PCA(n_components=900,whiten=True)
batch_size=32
path='ModelNet10pcd'
classes=os.listdir(path)
true_label=[]
true_data=[]
validation_data=[]
train_data=[]
train_label=[]
validation_label=[]
print classes
num_class=len(classes)
n=20
N=900
# validation_size=N*n/10
for file in classes:
  train_path='train_fisher/'+file+'_fisher_train.txt'
  data=np.loadtxt(train_path,delimiter=',')
  # data=StandardScaler().fit_transform(data)
  shape=data.shape
  b=data
  while b.shape[0]<N-shape[0]:
  	b=np.append(b,data,axis=0)
  x=N-b.shape[0]
  b=np.append(b,data[:x],axis=0)
  print b.shape
  b=shuffle(b)

  # b=StandardScaler().fit_transform(b)
  label=np.zeros(num_class)
  index=classes.index(file)
  label[index]=1.0
  val=N*20/100
  nval=N-val
  for i in range(nval):
	train_label.append(label)
	train_data.append(b[i])
  for j in range(nval,N):
  	validation_data.append(b[i])
  	validation_label.append(label)
  # train_data.append(data[:100-validation_size])
  # validation_data.append(data[100-validation_size:])


# true_data=np.reshape(true_data,[tr*10,shape[1]])
# true_data=normalize(true_data, norm='l2')
# pca.fit_transform(true_data)
train_data=StandardScaler().fit_transform(train_data)
validation_data=StandardScaler().fit_transform(validation_data)
train_data=np.array(train_data)
validation_data=np.array(validation_data)
train_label=np.array(train_label)
validation_label=np.array(validation_label)
train_data=train_data[:,np.newaxis,:]
validation_data=validation_data[:,np.newaxis,:]
train_data,train_label=shuffle(train_data,train_label)
validation_data,validation_label=shuffle(validation_data,validation_label)
print train_data.shape
print validation_data.shape
print train_label.shape
#train

# val=true_data.shape[0]*20/100


class DataSet(object):

  def __init__(self, images, labels):
	self._num_examples = images.shape[0]

	self._images = images
	self._labels = labels
	self._epochs_done = 0
	self._index_in_epoch = 0

  @property
  def images(self):
	return self._images

  @property
  def labels(self):
	return self._labels


  @property
  def num_examples(self):
	return self._num_examples

  @property
  def epochs_done(self):
	return self._epochs_done

  def next_batch(self, batch_size):
	"""Return the next `batch_size` examples from this data set."""
	start = self._index_in_epoch
	self._index_in_epoch += batch_size

	if self._index_in_epoch > self._num_examples:
	  # After each epoch we update this
	  self._epochs_done += 1
	  start = 0
	  self._index_in_epoch = batch_size
	  assert batch_size <= self._num_examples
	end = self._index_in_epoch

	return self._images[start:end], self._labels[start:end]

def shapes():
  """docstring for ClassName"""
  size=train_data.shape

  return size
	
def read_train_sets():
  class DataSets(object):
	pass
  data_sets = DataSets()

  

  
  validation_images = validation_data
  validation_labels = validation_label

  train_images =train_data
 
  train_labels = train_label

  data_sets.train = DataSet(train_images, train_labels)
  data_sets.valid = DataSet(validation_images, validation_labels)
  return data_sets