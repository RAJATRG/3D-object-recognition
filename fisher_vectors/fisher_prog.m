close all;
clc
cd /home/rajat ;
run('VLFEATROOT/toolbox/vl_setup');
path='/home/rajat/Desktop/Summer-Internship/week3/task1';
cd(path);
list=dir('ModelNet10pcd');
% num_cluster=input('input-data\n');
for i=3:length(list)
%     train
    s=strcat('ModelNet10pcd/',list(i).name,'/train');
    x=strcat(s,'/*txt');
    txt_list=dir(x)
    fisher=[];
    for j=1:length(txt_list)
        text_path=strcat(s,'/',txt_list(j).name);
        descriptors=dlmread(text_path,'',11,0);
        [means,covariance,prob]=vl_gmm(descriptors',num_cluster);
        fisher(j,:)=(vl_fisher(descriptors',means,covariance,prob))';
    end
    size(fisher)
    size(txt_list)
    fisher_path=strcat(s,'/',list(i).name,'_fisher_train.txt');
    csvwrite(fisher_path, fisher);
%     test
    s=strcat('ModelNet10pcd/',list(i).name,'/test');
    x=strcat(s,'/*txt');
    txt_list=dir(x);
    fisher=[];
    for k=1:length(txt_list)
        text_path=strcat(s,'/',txt_list(k).name);
        descriptors=dlmread(text_path,'',11,0);
        [means,covariance,prob]=vl_gmm(descriptors',num_cluster);
        fisher(k,:)=(vl_fisher(descriptors',means,covariance,prob))';
    end
    size(fisher)
    size(txt_list)
    fisher_path=strcat(s,'/',list(i).name,'_fisher_test.txt');
    csvwrite(fisher_path,fisher);
end

        
    