function bilinear_zoom(fac)

im0=imread('20170115145041.jpg');
im=cast(im0,'int16'); 
imshow(cast(im,'uint8')); 
pause;

[h,v,d]=size(im);

for i=1:h
    for j=1:v
      im1(1+(i-1)*fac,1+(j-1)*fac,:)=im(i,j,:);  
    end
       imshow(cast(im1,'uint8')); 
end


for i=1:1+(h-2)*fac     
    for j=1:1+(v-2)*fac 
    
       if ((rem(i-1,fac)==0) && (rem(j-1,fac)==0))   
       else  
           h00=im1( ceil(i/fac)*fac-fac+1,ceil(j/fac)*fac-fac+1,:); 
           h10=im1( ceil(i/fac)*fac-fac+1+fac,ceil(j/fac)*fac-fac+1,:);
           h01=im1( ceil(i/fac)*fac-fac+1,ceil(j/fac)*fac-fac+1+fac,:);
           h11=im1( ceil(i/fac)*fac-fac+1+fac,ceil(j/fac)*fac-fac+1+fac,:);
           
           x=rem(i-1,fac); 
           y=rem(j-1,fac);  
          
           dx=x/fac; 
           dy=y/fac;
          
           b1=h00;    
           b2=h10-h00;
           b3=h01-h00;
           b4=h00-h10-h01+h11;           
           im1(i,j,:)=b1+b2*dx+b3*dy+b4*dx*dy; 
         end
        end
  imshow(cast(im1,'uint8')); 
end

imshow(cast(im1,'uint8'));

