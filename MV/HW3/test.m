a = imread('test.tif');
% figure (1);
% imshow(a);


% se = ones(70);
fe = imerode(a, strel('line', 71 , 0));
fo = imopen(a,strel('line',71 , 0));
b = imreconstruct(a , fe);
% figure (2);
% imshow(b);

c =imopen(b ,strel('line' , 72 , 0));

c = imopen(c,strel('rectangle' , [20 20]));
% figure (3);
% imshow(c);

se = strel('disk',55);
% d =imopen(a ,se);
d= a - b;
% d1 = imtophat(c,se)
% % d=a-c;
% d=d-d1;
d1 = imreconstruct(a , d);
% figure (7);
% imshow(d1);

d=imerode(a, ones(1,71));
d = imreconstruct(d,a); 
d = imsubtract(a, d); 
% figure (4);
% imshow(d);

ffe = imerode(d , strel('line' , 10 , 0));
e = imopen(a ,strel('line' , 10 , 0));
ec = imreconstruct(ffe, d);
ff= d-ec;
% figure (5);
% imshow(ec);

SE = strel('line' , 20 , 0);
g= imdilate(ec,strel('line' , 30 , 0));
% BW = im2bw(ff,0.3);
g1= g - ff;
% figure (6);
% imshow(g1);

h= min(d , g);
h = imreconstruct(h, d);
% figure (7);
% imshow(h);
% figure(8), imshow(h)
    
j = im2bw(h, 0.6); 
% figure(10), imshow(j)
i = im2bw(h, 0.3); 
% figure(9), imshow(i)



temp1= imdilate(j, ones(1,31)) ;
large = imreconstruct(temp1 ,i);

large_keys = imdilate(large, ones(13,31));
% figure(11), imshow(large_keys)
% 

temp2=i - large; 
small = im2bw(temp2); 
small_keys = imdilate(small, ones(5, 31));
% figure(12), imshow(small_keys)


figure(13)
subplot(3,3,1)
figure(13), imshow(a)
title('a :orginal image')
subplot(3,3,2)
figure(13), imshow(b)
title('b : Opening by reconstruction of (a)')
figure (13)
subplot(3,3,3)
figure(13), imshow(c)
title('c : Opening of (a)')
subplot(3,3,4)
figure(13), imshow(d)
title('d : Top-hat by reconstruction ')
subplot(3,3,5)
figure(13), imshow(d1)
title('e : a top-hat transformation.')
subplot(3,3,6)
figure(13), imshow(ec)
title('f : Opening by reconstruction of (d)')
subplot(3,3,7)
figure(13), imshow(g1)
title('g : Dilation of (f)')
subplot(3,3,8)
figure(13), imshow(h)
title('h : Minimum of (d) and (g)')
subplot(3,3,9)
figure(13), imshow(i)
title('i : threshold ')




figure(14), imshow(i)
Lreg = regionprops(large_keys,'Area', 'BoundingBox'); Sreg = regionprops(small_keys, 'Area', 'BoundingBox');keys = cat(1, Lreg.Area);words = cat(1, Sreg.Area);
max_keys = max(keys);min_keys = min(keys);max_words = max(words);min_words = min(words);
for k = 1 : length(Lreg)
  BoundingBox = Lreg(k).BoundingBox; 
  if Lreg(k).Area == max_keys
      rectangle('Position', [BoundingBox(1)-5,BoundingBox(2)-5,BoundingBox(3)+10,BoundingBox(4)+10],...
      'EdgeColor','b','LineWidth',4)
  elseif Lreg(k).Area == min_keys
      rectangle('Position', [BoundingBox(1)-5,BoundingBox(2)-5,BoundingBox(3)+10,BoundingBox(4)+10],...
      'EdgeColor','r','LineWidth',1)
  else
      rectangle('Position', [BoundingBox(1)-5,BoundingBox(2)-5,BoundingBox(3)+10,BoundingBox(4)+10],...
      'EdgeColor','g','LineWidth',2)
  end
end
for k = 1 : length(Sreg)
  BoundingBox = Sreg(k).BoundingBox;
  if Sreg(k).Area == max_words
      rectangle('Position', [BoundingBox(1)-5,BoundingBox(2)-5,BoundingBox(3)+10,BoundingBox(4)+10],...
      'EdgeColor','b','LineWidth',4)
  elseif Sreg(k).Area == min_words
      rectangle('Position', [BoundingBox(1)-5,BoundingBox(2)-5,BoundingBox(3)+10,BoundingBox(4)+10],...
      'EdgeColor','r','LineWidth',1)
  else
      rectangle('Position', [BoundingBox(1)-5,BoundingBox(2)-5,BoundingBox(3)+10,BoundingBox(4)+10],...
      'EdgeColor','y','LineWidth',2)
  end
end



