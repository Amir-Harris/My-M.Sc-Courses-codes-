function [image] = myScaleogram(x,scales,b)
%% computes scaleogram

interval = -5*pi:0.1:5*pi;
slen = length(scales);
xlen = length(x);
image = zeros(slen,xlen);

for i=slen:-1:1
    y = 2 * sinc(2*interval/scales(i)) - sinc(interval/scales(i));
    image(i,:) = conv(x,y,'same');
    
end

if(b)
    figure;
    imshow(image,[]);
end