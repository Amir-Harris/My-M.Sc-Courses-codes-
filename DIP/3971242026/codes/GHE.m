clear;
close all;
img =  imread('butterfly_noisy.bmp');

HE = zeros(256 , 2);
HE(1:256) = 0:255;
for i=1:256
    HE(i,2) =  numel(find(img == i-1));
end


HE(:,3) = HE(:,2) ./ numel(img);
HE(:,4) = HE(:,3) .* 255;

for i=1:256
    HE(i,5) = sum(HE(1:i,4));
end

HE(:,6) = round(HE(:,5));
mapEQCoins =  [HE(:,1) , HE(:,6)];

normalimg = uint8(zeros(size(img)));
for i=1:256
    findIndex  = find(img == mapEQCoins(i,1));
    if(~isempty(findIndex))
        normalimg(findIndex) = mapEQCoins(i,2);
    end
end

hsnormalimg = zeros(256 , 2);
hsnormalimg(1:256) = 0:255;
for i=1:256
    hsnormalimg(i,2) =  numel(find(normalimg == i-1));
end


figure ;
imshow(img);
title('original image');
figure (2) ;
imshow(normalimg);
title('global histogram equalization ');

