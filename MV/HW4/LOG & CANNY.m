I=imread('home.tif');



%% LOG

Final = edge(I,'log',0.5,0.3);
figure(2);
imshow(Final)


%% new approach of LOG


% w=fspecial('log',[7 7],0.3); 
% filtered_img= imfilter(I,w,'replicate'); 
% j = im2Final(filtered_img, 0.5);
% figure (1);
% imshow(j);



% imshowpair(j,Final,'montage')    
% title('new approach                                                                old approach');


%%  CANNY 

% Final2 = edge(I,'CANNY',0.3 ,0.5);
% figure (1);
% imshow(Final2);
% title('SIGMA = 10.5 & THRESHOLD = 0.3 ');    



%%  print it as movie

% movie2 = VideoWriter('dilationreconstruction','MPEG-4' );
% open(movie2)cc
% 
% imshow(Final2)
% drawnow
% F = getframe(gcf);
% writeVideo(movie2,F);
% 
% close(movie2);
