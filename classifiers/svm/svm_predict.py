import numpy as np 
import os
from sklearn.externals import joblib
from sklearn import svm
from sklearn.metrics import accuracy_score
clfp= joblib.load('filename.pkl')
path='ModelNet10pcd'
classes=os.listdir(path)
x_test=[]
y_true=[]
print classes
for file in classes:
	test_path=file+'/test'+'/'+file+'_fisher_test.txt'
	data=np.loadtxt(path+'/'+test_path,delimiter=',')

	y_true=y_true+[file]*50
	data=data.tolist()
	for i in xrange(50):
	 	x_test.append(data[i])
	print file+ '_Done'


x_test=np.array(x_test)
y_true=np.array(y_true)
y=clfp.predict(x_test)
print accuracy_score(y_true,y)*100,'%'