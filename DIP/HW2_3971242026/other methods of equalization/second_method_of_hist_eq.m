%Read a grayscale Image or a matrix mxn

A=imread('cameraman.png');


subplot(1,2,1); imshow (A);title ('orginal');
subplot(1,2,2); imhist (A);

%Specify the bin range[0 255]

bin=255;

%Find the histogram of the image.

Val=reshape(A,[],1);

Val=double(Val);

I=hist(Val,0:bin);


%Divide the result by number of pixels

Output=I/numel(A);


%Calculate the Cumlative sum

CSum=cumsum(Output);



%Perform the transformation S=T(R) where S and R in the range [ 0 1]

HIm=CSum(A+1);


%Convert the image into uint8

HIm=uint8(HIm*bin);


figure (2);
subplot(1,2,1); imshow (HIm);title ('hist');
subplot(1,2,2); imhist (HIm);