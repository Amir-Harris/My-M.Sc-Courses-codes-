clc ;
clear all;

img = imread ('4.jpg');
x= double (img);
imshow(x/255);
c0 = mod(x,2);
c1 = mod(floor(x/2),2);
c2 = mod(floor(x/4),2);
c3 = mod(floor(x/8),2);
c4 = mod(floor(x/16),2);
c5 = mod(floor(x/32),2);
c6 = mod(floor(x/64),2);
c7 = mod(floor(x/128),2);

figure;
set (gcf,'position', get (0,'screensize'));
subplot(241); imshow(c0); title ('bit 0');
subplot(242); imshow(c1); title ('bit 1');
subplot(243); imshow(c2); title ('bit 2');
subplot(244); imshow(c3); title ('bit 3');
subplot(245); imshow(c4); title ('bit 4');
subplot(246); imshow(c5); title ('bit 5');
subplot(247); imshow(c6); title ('bit 6');
subplot(248); imshow(c7); title ('bit 7');

