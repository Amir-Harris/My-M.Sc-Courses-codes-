function [spec] = mySpectrogram(x,window,b )
%% computes spectrogram

img = Hn( x , window );

[m,n] = size(img);
spec = zeros(m,n);

for i=1:n
    spec(:,i) = myDFT(img(:,i));
end

spec = abs(spec);

if(b)
    figure;
    imshow(spec',[]);
end

end

