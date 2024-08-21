%C:\Users\SAMA LAPTOP\ML - Final tamrins\som\Matlabyar% self organizing map Kohonen
clc
clear all
% close all
n1=3;
%x=rand(2,1000);
load('dataset.mat')


y = traindata;
m = mean(y,2).^5;
label=traindata(:,111);
x = [label m]';


x = y(:,108:109)';

%A = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,2,7,9,12,27,10,8,14,5,7,8,9,2,7,5,3,6,3,0,4,2,2,3,1,2,5,3,5,6,4,1,3,6,3,9,8,6,13,17,11,23,17,8,20,14,10,14,12,11,19,12,12,20,18,18,19,21,13,10,19,36,24,21,28,30,15,44,32,32,47,36,37,59,44,40,87,79,100,307,453,951,10129,0]
%C = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,6,3,7,15,23,24,36,33,29,46,35,45,50,27,22,48,18,21,19,20,12,20,5,10,17,11,8,15,12,11,20,5,10,17,15,12,20,25,13,28,21,8,14,11,31,18,27,25,18,16,30,19,20,31,27,35,36,22,34,55,27,43,61,45,48,98,97,112,381,568,1428,8980,0]
%B = mean(C)

%x=traindata;
%meanData = sum(x,2)

%sum = 0
%meanData = zeros(1200,2)
%for i=1:1200
%    for j=1:110
%        meanData(i,j) = round(mean(x(1:110,i),2))
%    end
%end

%meanData = round(mean(x,3))
    

w=rand(2,n1,n1)*(0.52-0.48)+0.48;
%// learning rate (updates by every epoch)
lRInit = 1;

figure(3)
plot(x(1,:),x(2,:),'*b')
hold on
plot(reshape(w(1,:,:),size(w,2),size(w,3)),reshape(w(2,:,:),size(w,2),size(w,3)),'or')
%plot(reshape(w(1,:,:),size(w,2),size(w,3)),reshape(w(2,:,:),size(w,2),size(w,3)),'k','linewidth',2)
%plot(reshape(w(1,:,:),size(w,2),size(w,3))',reshape(w(2,:,:),size(w,2),size(w,3))','k','linewidth',2)
hold off
title('t=0');
drawnow
do=4;
T=99;
t=1;
while (t<=T)
    lR=lRInit-(lRInit*t)/T;
    d=round(do*(1-t/T));
    %loop for the 1000 inputs
    for i=1:1200
        %Euclidean distance
        dist(:,:) = sqrt(sum((w - repmat(x(:,i),1,n1,n1)).^2,1));
        
        [v minj1] = min(dist,[],1);
        [v minj2] = min(v,[],2);
        
        j1star= minj1(minj2);
        j2star= minj2;
        %update the winner neuron
        w(:,j1star,j2star)=w(:,j1star,j2star)+lR*(x(:,i)- w(:,j1star,j2star));
        %update the neighbour neurons
%         exp(-d*sqrt((2*d).^2)/(2*d));
        for dd=1:1:d
            jj1=j1star-dd;
            jj2=j2star;
            disBMU=exp(-2*dd/(2*d));
            if (jj1>=1)
                w(:,jj1,jj2)=w(:,jj1,jj2)+lR*disBMU*(x(:,i)- w(:,jj1,jj2));
                for k=1:d-1
                    jj2=j2star-k;
                    if (jj2>=1)
                        disBMU=exp(-sqrt((2*(k+d).^2)/(2*d)));
                        w(:,jj1,jj2)=w(:,jj1,jj2)+lR*disBMU*(x(:,i)- w(:,jj1,jj2));
                    end
                    jj2=j2star+k;
                    if (jj2<=3)
                        disBMU=exp(-sqrt((2*(k+d).^2)/(2*d)));
                        w(:,jj1,jj2)=w(:,jj1,jj2)+lR*disBMU*(x(:,i)- w(:,jj1,jj2));
                    end
                end
            end
            jj1=j1star+dd;
            jj2=j2star;
            disBMU=exp(-2*dd/(2*d));
            if (jj1<=3)
                w(:,jj1,jj2)=w(:,jj1,jj2)+lR*disBMU*(x(:,i)- w(:,jj1,jj2));
                for k=1:d-1
                    jj2=j2star-k;
                    if (jj2>=1)
                        disBMU=exp(-sqrt((2*(k+d).^2)/(2*d)));
                        w(:,jj1,jj2)=w(:,jj1,jj2)+lR*disBMU*(x(:,i)- w(:,jj1,jj2));
                    end
                    jj1=j1star+dd;
                    jj2=j2star+k;
                    if (jj2<=3)
                        disBMU=exp(-sqrt((2*(k+d).^2)/(2*d)));
                        w(:,jj1,jj2)=w(:,jj1,jj2)+lR*disBMU*(x(:,i)- w(:,jj1,jj2));
                    end
                end
            end
            disBMU=exp(-2*dd/(2*d));
            jj1=j1star;
            jj2=j2star-dd;
            if (jj2>=1)
                w(:,jj1,jj2)=w(:,jj1,jj2)+lR*disBMU*(x(:,i)- w(:,jj1,jj2));
            end
            jj1=j1star;
            jj2=j2star+dd;
            if (jj2<=3)
                w(:,jj1,jj2)=w(:,jj1,jj2)+lR*disBMU*(x(:,i)- w(:,jj1,jj2));
            end
        end
    end
    t=t+1;
    figure(3)
    plot(x(1,:),x(2,:),'*b')
    hold on
    plot(reshape(w(1,:,:),size(w,2),size(w,3)),reshape(w(2,:,:),size(w,2),size(w,3)),'or')
    %plot(reshape(w(1,:,:),size(w,2),size(w,3)),reshape(w(2,:,:),size(w,2),size(w,3)),'k','linewidth',2)
    %plot(reshape(w(1,:,:),size(w,2),size(w,3))',reshape(w(2,:,:),size(w,2),size(w,3))','k','linewidth',2)
    hold off
    title(['Epoch no.' num2str(t)]);
    drawnow
    
end

