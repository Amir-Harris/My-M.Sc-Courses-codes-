function [img] = Hn(x,window)
%% creates an image using overlaped segments of signal

xlen = length(x);

img = zeros(window,xlen-window+1);

[m,~] = size(x);

if( m==1 )
    x = x';
end

% stack segments to create the image
for i=1:(xlen-window+1)
    img(:,i) = x(i:i+window-1);
end

end
