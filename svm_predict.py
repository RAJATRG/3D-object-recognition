import numpy as np 
import os
from sklearn.externals import joblib
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import normalize
clfp= joblib.load('filename.pkl')
path='ModelNet10pcd'
classes=os.listdir(path)
acc=0
x_test=[]
y_true=[]
N=50
# pca=PCA(n_components=908,whiten=False)
print classes
for file in classes:
	test_path='test_fisher/'+file+'_fisher_test.txt'
	data=np.loadtxt(test_path,delimiter=',')
	y_true=y_true+[file]*data.shape[0]
	for i in xrange(data.shape[0]):
	 	x_test.append(data[i])
	print file+ '_Done'
	





x_test=np.array(x_test)

x_test=StandardScaler().fit_transform(x_test)
# x_test=pca.fit_transform(x_test)
# x_test=normalize(x_test, norm='l2')
print x_test.shape
y_true=np.array(y_true)
y=clfp.predict(x_test)
accuracy_=accuracy_score(y_true,y)
print '_accuracy',accuracy_*100,'%'

# print 'Total_accuracy',acc*10.0,'%'

