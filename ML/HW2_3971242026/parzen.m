clc;
clear all;
close all;
%%%%%%%%%%%%%%%%%%%% Definition Matrix %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
load Hist_dataset.mat;
dataset=traindata;
[row,col]=size(dataset);
 
for i=1:col-1
    Min=min(dataset(:,i));
    Max=max(dataset(:,i));
    dataset(:,i)= (dataset(:,i) -Min)/(Max - Min);
end
%[row,col]=size(dataset);
dataset(:,1:8)=[];
[row1,col1]=size(dataset);

number_of_class=12;

%k=3;
r=1.02;
%-------------------------------select train and test dataset
train=zeros(floor(0.7*row),col1);
test=zeros(ceil(0.3*row) ,col1);

m2=row1;
for i=1:floor(0.7*row1)
    r=(randi(m2-1)+1);
    train(i,:)=dataset(r,:);
    dataset(r,:)=[];
    m2=m2-1;
end
test=dataset;

[TrainCnt,ee]=size(train);
[TestCnt,n]=size(test);
PredicttClass=zeros(TestCnt,1);

%[row1,col1]=size(train);
%col1
%row


%-------------------------------------- knn algoritm

for i=1:TestCnt
    dist=zeros(TrainCnt,1);
     dist2=zeros(TrainCnt,2);
    for j=1:TrainCnt 
        sum=0;
        for n=1:col1-1
            sum=((test(i,n)-train(j,n))^2)+sum;
        end
        dist(j,1)= sqrt(sum);

        %dist(j,1)= sqrt(((test(i,1)-train(j,1))^2)+((test(i,2)-train(j,2))^2) + ((test(i,3)-train(j,3))^2)+((test(i,4)-train(j,4))^2));
        dist2(j,1)=dist(j,1);
        dist2(j,2)=j;
    end
    dist=sort(dist);
    CntofClass=zeros(number_of_class,1);
    a(1,1)=0;
    for n=1:size(dist) 
        
        if dist(n,1)<=r
         a=a+1;      
        end  
        
    end
    k=a;
     for z=1:k
       Nearest=find(dist2(:,1)==dist(z));
       ClsN=train(Nearest,col1);
       ClsN1=ClsN+1;
       
       CntofClass(ClsN1)=CntofClass(ClsN1)+1;
    end
    
    class=find(CntofClass==max(CntofClass));
    class1(1)=class(1)-1;
    PredicttClass(i)=class1(1);
    
end
error=test(:,col1)-PredicttClass;
[Miss]=find(error~=0);
[Miss1]=find(error==0);
k=k
Error =100*size(Miss,1) /TestCnt
per=100*size(Miss1,1) /TestCnt

