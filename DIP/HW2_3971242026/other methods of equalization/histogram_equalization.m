i = imread ('cameraman.png');
figure;
subplot(1,2,1); imshow (i);title ('orginal');
subplot(1,2,2); imhist (i);
figure (2);
ih = histeq (i);
subplot(2,2,1); imshow (ih);title ('equalized');
subplot(2,2,2); imhist (ih);