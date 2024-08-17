clc;
close all;
clear all;

% Read the Image 
% Display the original Image

image = imread('cameraman.png');
subplot(3,3,1); 
imshow(image); title('Original Image');
image = double(image);
% the calculation

kernelx = [ -1, 0, 1;
            -2, 0, 2;
            -1, 0, 1];

kernely = [  1, 2, 1;
             0, 0, 0;
            -1,-2,-1];

height = size(image,1);
width = size(image,2);
channel = size(image,3);

for i = 2:height - 1
    for j = 2:width - 1
        for k = 1:channel
            magx = 0;
            magy = 0;
            for a = 1:3
                for b = 1:3
                    magx = magx + (kernelx(a, b) * image(i + a - 2, j + b - 2, k));
                    magy = magy + (kernely(a, b) * image(i + a - 2, j + b - 2, k));
                end;
            end;     
            edges(i,j,k) = sqrt(magx^2 + magy^2); 
        end;
    end;
end;
edges = uint8(edges);



% Apply Sobel Operator
% Display only the horizontal Edges

sobelhz = edge(image,'sobel','horizontal');
subplot(2,3,2); 
imshow(sobelhz,[]); title('Sobel - Horizontal Edges');

% Apply Sobel Operator
% Display only the vertical Edges

sobelvrt = edge(image,'sobel','vertical');
subplot(2,3,3); 
imshow(sobelhz,[]); title('Sobel - Vertical Edges');


% Apply Sobel Operator
% Display both horizontal and vertical Edges
sobelvrthz = edge(image,'sobel','both');
subplot(2,3,4); 
imshow(sobelvrthz,[]); title('Sobel - All edges');




