img = imread('strawberry.jpg');

red_channel = img(:,:,1);
green_channel = img(:,:,2);
blue_channel = img(:,:,3);         


red_hist = histeq(red_channel);     
green_hist = histeq(green_channel);
blue_hist = histeq(blue_channel);

comb_hist = cat(3,red_hist,green_hist,blue_hist);   
comb_red_hist = cat(3,red_hist,green_channel,blue_channel); 
comb_green_hist = cat(3,red_channel,green_hist,blue_channel); 
comb_blue_hist = cat(3,red_channel,green_channel,blue_hist); 

figure(1);imshow(comb_hist);title('combination');

figure(2);imshow('strawberry.jpg'); title('Orginal');

figure;
subplot(2,3,1);
imshow(red_channel ,[]);
title('red channel');
subplot(2,3,2);
imshow(green_channel ,[]);
title('green channel');
subplot(2,3,3);
imshow(blue_channel,[]);
title('blue channel');
subplot(2,3,4);
imshow(comb_red_hist,[]);
title('red hist');
subplot(2,3,5);
imshow(comb_green_hist,[]);
title('green hist');
subplot(2,3,6);
imshow(comb_blue_hist,[]);
title('blue hist');