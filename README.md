# 3D-object-recognition
NIT-IITD Internship

downlaod the dataset:
http://modelnet.cs.princeton.edu/
ModelNET10

In Modelnet2PCDMaster :
# run bash converter.sh

to convert .off to .pcd

Then run : 
# python main.py

For SVM :
for training
# python svm_fisher.py 
for prediction 

# python svm_predict.py


For Neural Netowork:

for training 
# python nn_modelnet.py

for prediction 

# python nn_modelnet_predict.py
