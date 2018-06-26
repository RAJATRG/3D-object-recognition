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
print classes
num_class=len(classes)
n=20
N=100
validation_size=N*n/10
for file in classes:
  train_path='train_fisher/'+file+'_fisher_train.txt'
  data=np.loadtxt(train_path,delimiter=',')
  shape=data.shape
  print shape
  data=data[:N]
  label=np.zeros(num_class)
  index=classes.index(file)
  label[index]=1.0
  for i in range(N):
    true_label.append(label)
  true_data.append(data)
  # train_data.append(data[:100-validation_size])
  # validation_data.append(data[100-validation_size:])


true_data=np.reshape(true_data,[N*10,shape[1]])
# true_data=normalize(true_data, norm='l2')
# pca.fit_transform(true_data)
true_data=StandardScaler().fit_transform(true_data)
true_label=np.array(true_label)
true_data=true_data[:,np.newaxis,:]
print true_data.shape
print true_label.shape
#train



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
  size=true_data.shape

  return size
    
def read_train_sets():
  class DataSets(object):
    pass
  data_sets = DataSets()

  images,labels =true_data,true_label 
  images,labels=shuffle(images,labels)

  
  validation_images = images[N*10-validation_size:]
  validation_labels = labels[N*10-validation_size:]

  train_images =images[:N*10-validation_size]
 
  train_labels = labels[:N*10-validation_size]

  data_sets.train = DataSet(train_images, train_labels)
  data_sets.valid = DataSet(validation_images, validation_labels)
  return data_sets