function [X] = myDFT(x)
%% computes dft of signal x
[m,n] = size(x);
X = zeros(m,n);

if( m==1 )
    N = n;
else
    N = m;
end

for i=0:(N-1)
    sum = 0;
    
    for j=0:(N-1)
        sum = sum + x(j+1)*exp(-(2*pi*1i/N)*i*j);
    end
    
    X(i+1) = sum;
end

end

