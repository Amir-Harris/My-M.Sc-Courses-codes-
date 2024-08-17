clc;
close all;
clear all;
img= imread('butterfly_noisy.bmp');
img= (img);

MyHist=imhist(uint8(img))';
m = zeros(1,5);
[R C] = size(img);
for i = 1 : 256
    if MyHist(i)>R*C/100000;     m(1) = i; break;
    end
end
for i = 256 :-1: 1
    if MyHist(i)>R*C/100000;     m(5) = i; break;
    end
end
data = cumsum(MyHist)/(R*C);
[val idx] = find(data >= 0.25 &data < 0.50);
if size(idx) ~= 0
    m(2) = idx(1);
    m(3) = idx(end);
else
    [val idx] = find(data > 0);
    m(2) = m(1);
    m(3) = m(1);
end
[val idx] = find(data >= 0.75);
m(4) = idx(1)-1;
avgFreqOfimage = floor(R*C/256);
X = zeros(1,256);
for i = 1 :256
    if MyHist(i) >= avgFreqOfimage
        X(i) = avgFreqOfimage; 
    else 
    X(i) = MyHist(i);
    end
end
span= zeros(1,4);
for i = 1 : 4
    span(i) = m(i+1)-m(i);
end
range = round(256*span/sum(span));
sepPosition = cumsum([0 range]);
Y= zeros(1,256);
map = zeros(1,256);
for i = 1 : 4
     if i==1      
         temp = cumsum(X(m(i):m(i+1)));
        Y(m(i):m(i+1)) = temp/temp(end);
        map(m(i):m(i+1)) = round(range(i)*Y(m(i):m(i+1))); 
     else
         temp = cumsum(X(m(i):m(i+1)));
        Y(m(i):m(i+1)) = temp/temp(end);
        map(m(i):m(i+1)) = round(range(i)*Y(m(i):m(i+1))+sepPosition(i));        
    end   
end
final = zeros(R,C);
for i = 1: R
    for j = 1: C
        final(i,j) = map(1 + int16(img(i,j)));
    end
end
final = uint8(final);
        figure(1), imshow(final)
        title('BPDHE image');
        figure(2), imshow(img)
        title('original image');