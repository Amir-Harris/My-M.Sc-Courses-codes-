clc ;
clear all;

img = imread ('4.jpg');
x = rgb2gray(img);

[m,n]=size (x);

bit = zeros (m,n,8);

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



% showing img

out = 255 * bit ;
out = uint8 (out);

figure;
set (gcf,'position', get (0,'screensize'));
subplot(241); imshow(out (:,:,1)); title ('bit 0');
subplot(242); imshow(out (:,:,2)); title ('bit 1');
subplot(243); imshow(out (:,:,3)); title ('bit 2');
subplot(244); imshow(out (:,:,4)); title ('bit 3');
subplot(245); imshow(out (:,:,5)); title ('bit 4');
subplot(246); imshow(out (:,:,6)); title ('bit 5');
subplot(247); imshow(out (:,:,7)); title ('bit 6');
subplot(248); imshow(out (:,:,8)); title ('bit 7');

figure (2);
imshow(x);title('org');



