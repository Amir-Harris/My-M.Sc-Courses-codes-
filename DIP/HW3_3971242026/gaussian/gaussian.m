% Flush out the MATLAB.
close all;
clc;
clear all;


% Read the desired image file.
ImageData = imread('cameraman.png');




% Alter Original Image.
PaddedImage = double(ImageData);


% Take the input of the sigma from user.
SigmaValue = 2;


% Set the window size.
WindowSize = 4;

[X,Y]=meshgrid(-WindowSize:WindowSize,-WindowSize:WindowSize);

M = size(X,1)-1;

N = size(Y,1)-1;

Temp = -(X.^2+Y.^2)/(2*SigmaValue*SigmaValue);

FinalCalculation= exp(Temp)/(2*pi*SigmaValue*SigmaValue);


% Initialize and pad the output image.
OutputImage=zeros(size(PaddedImage));

PaddedImage = padarray(PaddedImage,[WindowSize WindowSize]);


% Perform convolution.
for i = 1:size(PaddedImage,1)-M
    
    for j =1:size(PaddedImage,2)-N
        Temp = PaddedImage(i:i+M,j:j+M).*FinalCalculation;
        
        OutputImage(i,j)=sum(Temp(:));
    end
    
end


% Display the output image.
OutputImage = uint8(OutputImage);
% Display the original image.
figure,imshow(ImageData);
title(' Original Image: ');

figure (2),imshow(OutputImage);
title(' Final Image: ');