import os
import sys
import glob
classes=os.listdir('/home/rajat/Desktop/Summer-Internship/week3/task1/ModelNet10pcd')
print classes
for file in classes:
	# train
	train_path=file+'/train'
	

	path = os.path.join('/home/rajat/Desktop/Summer-Internship/week3/task1/ModelNet10pcd', train_path, '*pcd')
	files=glob.glob(path)
	print len(files)
	for i in files:
		os.system("cd /home/rajat/Desktop/Summer-Internship/week3/shot/build && ./shot "+i+" "+i+".txt")
 # test
	train_path=file+'/test'
	path = os.path.join('/home/rajat/Desktop/Summer-Internship/week3/task1/ModelNet10pcd', train_path, '*pcd')
	files=glob.glob(path)
	print len(files)
	for i in files:
		os.system("cd /home/rajat/Desktop/Summer-Internship/week3/shot/build && ./shot "+i+" "+i+".txt")


os.system('matlab -r fisher_prog')