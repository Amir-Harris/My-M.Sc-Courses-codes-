clear all
close all
clc
f=imread('cameraman.png');
f=double(f);
figure,imshow(f,[]);title('org Image');
 
M=size(f,1); N=size(f,2); 
C=3; D=3; 
P=M+C-1; Q=N+D-1; 
fp=zeros(P,Q); 
fp(1:M,1:N)=f; 
hp=zeros(P,Q); 
hp(1,1)=-4; hp(2,1)=1; hp(1,2)=1; 
hp(P,1)=1; hp(1,Q)=1; 
Fp=fft2(double(fp), P, Q); 
Hp=fft2(double(hp), P, Q); 
H = fftshift(Hp);
F1 = abs(H); % Get the magnitude
F1 = log(F1+1); % Use log, for perceptual scaling, and +1 since log(0) is undefined
F1 = mat2gray(F1,[0 1]);%converting the matrix to the intensity image 

 
Gp=Fp .* Hp; % Product of FFTs
gp=ifft2(Gp); % Inverse FFT
gp=real(gp); % Take real part
g=gp(1:M, 1:N);
figure(2),imshow(g,[]);title('gray scale Image');
 
gnorm = g;
gshar = double(f) - gnorm;
figure(3),imshow(uint8(gshar));title('final Image');

