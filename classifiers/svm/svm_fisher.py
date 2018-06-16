import numpy as np 
import os
from sklearn.externals import joblib
from sklearn import svm
path='ModelNet10pcd'
classes=os.listdir(path)
train_label=[]
train_data=[]
print classes
for file in classes:
	train_path=file+'/train'+'/'+file+'_fisher_train.txt'
	data=np.loadtxt(path+'/'+train_path,delimiter=',')

	train_label=train_label+[file]*100
	data=data.tolist()
	for i in xrange(100):
	 	train_data.append(data[i])
	print file+ '_Done'


train_data=np.array(train_data)
train_label=np.array(train_label)
print train_data.shape
print len(train_label)
# print train_label
# train=np.squeeze(train_data,axis=0).shape

clf=svm.SVC()
clf.fit(train_data,train_label)
print "training done"
print clf.score(train_data,train_label)*100,"%"
joblib.dump(clf, 'filename.pkl') 

