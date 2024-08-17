clc ;
clear all;

img = imread ('4.jpg');
x = rgb2gray(img);
img1 = imread ('2.jpg');
x1 = rgb2gray(img1);



[m,n]=size (x);
[m1,n1]=size (x1);


bit = zeros (m,n,8);
bit1 = zeros (m1,n1,8);


for i=1:m
    for j=1:n
        k=0;
        num = x(i,j);
        while (num>0)
            k=k+1;
            bit(i,j,k)=uint8(num/2) - uint8((num-1)/2);
            num = uint8((num-1)/2) ; 
        end
    end
end

for i1=1:m1
    for j1=1:n1
        k1=0;
        num1 = x1 (i1,j1);
        while (num1>0)
            k1=k1+1;
            bit1(i1,j1,k1)=uint8(num1/2) - uint8((num1-1)/2);
            num1 = uint8((num1-1)/2) ; 
        end
    end
end

% showing img

out = 255 * bit ;
out = uint8 (out);

out1 = 255 * bit1 ;
out1 = uint8 (out1);



figure (2);
imshow(x);title('first picture');
figure (4);
imshow(x1);title('second picture');

%reconstruction 
recon = zeros (m1,n1);
for q1=1:m1
    for r1=1:n1
       recon (q1,r1) = (2^7*bit1(q1,r1,4))+(2^6*bit1(q1,r1,3))+(2^5*bit1(q1,r1,2))+(2^4*bit1(q1,r1,1));
    end
end
recon = zeros (m,n);
for q=1:m
    for r=1:n
       recon (q,r) = (2^7*bit(q,r,8))+ (2^6*bit(q,r,7)) +(2^5*bit(q,r,6)) +(2^4*bit(q,r,5));
    end
end

recon = uint8 (recon);
figure (3);
imshow (recon);title ('new one');