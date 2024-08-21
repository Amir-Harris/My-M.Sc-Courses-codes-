Img = imread('strawberry.jpg');
Img_hsv = rgb2hsv(Img);
V = Img_hsv(:,:,3);
eq_V = histeq(V);
Img_hsv(:,:,3) = eq_V;
Img_eq = hsv2rgb(Img_hsv);
figure,imshow(Img_eq);title('HE');
figure,imshow(Img); title('Orginal');