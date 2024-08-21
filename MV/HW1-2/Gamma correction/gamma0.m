gamma=2.2;

for i=0:255;
    f=power((i+0.5)/256,1/gamma);
    N(i+1)=uint8(f*256-0.1);
end  
img=imread('g2.bmp');
img0=rgb2ycbcr(img);

Y=img0(:,:,1);
[x y]=size(Y);
for row=1:x
    for col=1:y
        for i=0:255
        if (Y(row,col)==i)
             Y(row,col)=N(i+1);
             break; 
        end
        end
    end
end
img0(:,:,1)=Y;
img1=ycbcr2rgb(img0);
figure(1),imshow(img),title('original');
figure(2),imshow(img1),title('final');
