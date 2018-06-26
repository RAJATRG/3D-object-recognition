import numpy as np 
import os
from sklearn.externals import joblib
from sklearn import svm
from sklearn.utils import shuffle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize

# pca=PCA(n_components=908,whiten=False)
path='ModelNet10pcd'
classes=os.listdir(path)
train_label=[]
train_data=[]
print classes
N=100
for file in classes:
	train_path='train_fisher/'+file+'_fisher_train.txt'
	data=np.loadtxt(train_path,delimiter=',')
	train_label=train_label+[file]*N
	for i in xrange(N):
	 	train_data.append(data[i])
	print file+ '_Done'


train_data=np.array(train_data)
print train_data.shape
train_data=StandardScaler().fit_transform(train_data)
train_label=np.array(train_label)
# train_data=pca.fit_transform(train_data)
# train_data=normalize(train_data, norm='l2')
print train_data.shape

# print train_label
# train=np.squeeze(train_data,axis=0).shape

clf=svm.SVC()
clf.fit(train_data,train_label)
print "training done"
print clf.score(train_data,train_label)*100,"%"
joblib.dump(clf, 'filename.pkl') 

