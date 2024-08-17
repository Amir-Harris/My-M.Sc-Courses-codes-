img=im2double(imread('cameraman.tif'));
 
 %initialization
 
[x,y]=size(img);
s=zeros(x,y);
xx=zeros(x,y);


for i=2:x-1
    for j=2:y-1
        xx(i,j)=img(i+1,j)+img(i-1,j)+img(i,j+1)+img(i,j-1)-(4*img(i,j));
    end
end
for i=2:x-1
    for j=2:y-1
         s(i,j)=img(i,j)-xx(i,j);
    end
end

smooth = imgradient(img);
%grad=fspecial('average',3);
%smooth=imfilter(img,grad);
%laplasian
img=im2double(img);
yy=zeros(x,y);

for i=2:x-1
    for j=2:y-1
        temp1(i,j)=img(i+1,j-1)-img(i-1,j-1)+2*(img(i+1,j)-img(i-1,j))-img(i-1,j+1)+img(i+1,j+1);
        temp2(i,j)=img(i-1,j+1)-img(i-1,j-1)+2*(img(i,j+1)-img(i,j-1))-img(i+1,j-1)+img(i+1,j+1);
        yy(i,j)=sqrt(temp1(i,j)^2+temp2(i,j)^2);
    end
end

%average filter
temp3=zeros(x,y);
temp4=zeros(x,y);
for i=1:x
    for j=1:y
        temp3(i,j)=smooth(i,j)*(-xx(i,j));
        temp4(i,j)=temp3(i,j)+img(i,j);
    end
end


%gamma correction

f=uint8((temp4.^(0.4))*255);

figure;
subplot(2,3,1);imshow(img);title('original');
subplot(2,3,2);imshow(yy);title('gradian ');
subplot(2,3,3);imshow(s);title('laplasian');
subplot(2,3,4);imshow(smooth);title('smooth ');
subplot(2,3,5);imshow(f);title(' final img');
