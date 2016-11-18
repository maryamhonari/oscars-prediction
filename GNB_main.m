
%% Implementing Gaussian Naive Bayes using MATLAB
clc
clear all
close all
%load allData
load data_foj_full
%load data_foj
Num_feat=87;
Num_class=2;
Num_test=401;
Mn=zeros(Num_feat,Num_class);
Co=zeros(Num_feat,Num_feat,Num_class);
Cm=zeros(Num_class,Num_class);
Confidence_Matrix=zeros(Num_class,Num_class);
Cv=zeros(Num_class,Num_class);
CCR=0;
Train_Data=Data_full(1:Num_feat,1:4570,:);
Test_Data=Data_full(1:Num_feat,4421:end,:);
for i=1:Num_class
    Mn(:,i)=mean(Train_Data(:,:,i)');
end
for i=1:Num_class
    Co(:,:,i)=cov(Train_Data(:,:,i)');
end
for i=1:Num_class
    for j=1:550
        [Maxi,Max1,Max2,sum]=NaiveBayse(Test_Data(:,j,i),Mn,Co);
        Cm(i,Maxi+1)=Cm(i,Maxi+1)+1;
        Confidence_Matrix(i,Maxi+1)=Confidence_Matrix(i,Maxi+1)+((Max1-Max2)/sum);
        Cv(i,Maxi+1)=Cv(i,Maxi+1)+1;
    end
    CCR=CCR+Cm(i,i);
end
CCR=(CCR/600);
for i=1:Num_class
    for j=1:Num_class
        if(Cv(i,j)~=0)
            Confidence_Matrix(i,j)=Confidence_Matrix(i,j)/Cv(i,j);
        end
    end
end
clear i j Maxi Max1 Max2 Cv sum

        

