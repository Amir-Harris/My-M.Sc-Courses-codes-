a = imread('cameraman.tif');
b = im2double(a);
[m,n] = size(b);
c = zeros(2*m,2*n);
[p,q] = size(c);

for i = 1:p
    for j = 1:q
        if i <= m && j<= n
            c(i,j) = b(i,j);
        else
            c(i,j) = 0;
        end
    end
end

d = zeros(p,q);

for i = 1:p
    for j = 1:q
        d(i,j) = c(i,j).*(-1).^(i + j);
    end
end

e = fft2(d);

[x,y] = freqspace(p,'meshgrid');
z = zeros(p,q);
for i = 1:p
    for j = 1:q
        z(i,j) = sqrt(x(i,j).^2 + y(i,j).^2);
    end
end

H = zeros(p,q);
for i = 1:p
    for j = 1:q
        if z(i,j) <= 0.4 
            H(i,j) = 1;
        else
            H(i,j) = 0;
        end
    end
end

 
h1 = e.*H;
h2 = ifft2(h1);
h3 = zeros(p,q);
for i = 1:p
    for j = 1:q
        h3(i,j) = h2(i,j).*((-1).^(i+j));
    end
end


out = zeros(m,n);
for i = 1:m
    for j = 1:n
        out(i,j) = h3(i,j);
    end
end




figure;
subplot(3,3,1);
imshow(b);title('original image');
subplot(3,3,2);
imshow(c);title('padded image');
subplot(3,3,3);
imshow(d);title('pre processed image for calculating DFT');
subplot(3,3,4);
imshow(e);title('2D DFT of the pre processed image');
subplot(3,3,5);
imshow(H);title('Low Pass Filter Mask');
subplot(3,3,6);
imshow(h1);title('Low passed output');
subplot(3,3,7);
imshow(h2);title('output image after inverse 2D DFT');
subplot(3,3,8);
imshow(h3);title('Post Processed image');
subplot(3,3,9);
imshow([b out]);title('input image                 output image');