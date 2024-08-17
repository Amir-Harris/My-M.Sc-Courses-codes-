im1=imread('cameraman.png');


im1 = imread('cameraman.png');
im2 = imread('20170115145041.jpg');




numofpixels=size(im1,1)*size(im1,2);


figure,imshow(im1);

title('Original Image');
HIm=uint8(zeros(size(im1,1),size(im1,2)));

freq=zeros(256,1);

probf=zeros(256,1);

probc=zeros(256,1);

cum=zeros(256,1);

output=zeros(256,1);


%freq counts the occurrence of each pixel value.

%The probability of each occurrence is calculated by probf.


for i=1:size(im1,1)

    for j=1:size(im1,2)

        value=im1(i,j);

        freq(value+1)=freq(value+1)+1;

        probf(value+1)=freq(value+1)/numofpixels;

    end

end


sum=0;

no_bins=255;


%The cumulative distribution probability is calculated. 

for i=1:size(probf)

   sum=sum+freq(i);

   cum(i)=sum;

   probc(i)=cum(i)/numofpixels;

   output(i)=round(probc(i)*no_bins);

end

for i=1:size(im1,1)

    for j=1:size(im1,2)

            HIm(i,j)=output(im1(i,j)+1);

    end

end
M = zeros(256,1,'uint8'); 
hist1 = imhist(im1); 
hist2 = imhist(im2);
cdf1 = cumsum(hist1) / numel(im1); %compute the cdf of im1
cdf2 = cumsum(hist2) / numel(im2); %compute the cdf of im2
 
for idx = 1 : 256
    diff = abs(cdf1(idx) - cdf2);
    [~,ind] = min(diff);
    M(idx) = ind-1;
end
 
 
%matching here
out = M(double(im1)+1);
 
subplot(2,3,1),imshow(im1);
title('cameraman');
subplot(2,3,2),imshow(im2);
title('barbara');
subplot(2,3,3),imshow(out);
title('Histogram matched cameraman');
subplot(2,3,4),imhist(im1);
title('Histogram of cameraman');
subplot(2,3,5),imhist(im2);
title('Histogram of barbara');
subplot(2,3,6),imhist(out);
title('Histogram of matched image');
HIm = histeq (HIm);
figure (2);
subplot(1,2,1); imshow (HIm);title ('hist');
subplot(1,2,2); imhist (HIm);