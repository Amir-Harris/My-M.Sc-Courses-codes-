%% Load  image
clear all;
clc;
img = imread('coins.png');

%% Find edges
img = img(:,:,1);
figure();
subplot(121);
imshow(img);
title('Original Image');
gaussianFilter = fspecial('gaussian',20, 10);
img_filted = imfilter(img, gaussianFilter,'symmetric');
subplot(122);
imshow(img_filted);
title('Filted Image');
filted_edges = edge(img_filted, 'Canny');
figure();
subplot(121);
imshow(filted_edges);
title('Edges found in filted image');
img_edges = edge(img, 'Canny');
subplot(122);
imshow(img_edges);
title('Edges found in original image')
[H, theta, rho] = hough_acc(filted_edges); 

%% Plot/show accumulator array H
figure();
imshow(imadjust(mat2gray(H)),'XData',theta,'YData',rho,...
      'InitialMagnification','fit');
title('Hough transform');
xlabel('\theta'), ylabel('\rho');
axis on, axis normal, hold on;
colormap(hot);
peaks = hough_peaks(H, 5); 
%% Highlight peak locations on accumulator array
imshow(imadjust(mat2gray(H)),'XData',theta,'YData',rho,'InitialMagnification','fit');
title('Hough transform with peaks found');
xlabel('\theta'), ylabel('\rho');
axis on, axis normal, hold on;
plot(theta(peaks(:,2)),rho(peaks(:,1)),'o','LineWidth',3,'color','red');

%% Circle hough
figure(5)
imshow(img)
[centers, radii, metric] = imfindcircles(img,[15 30]);
centersStrong5 = centers(1:5,:); 
radiiStrong5 = radii(1:5);
metricStrong5 = metric(1:5);
viscircles(centersStrong5, radiiStrong5,'EdgeColor','r');
