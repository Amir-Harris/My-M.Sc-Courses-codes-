



%   FUZZY_MEMBERSHIP_TYPE:
%           
%       triangular - uses a triangular membership function
%       gaussian - uses a gaussian membership function
%       custom - uses the  user defined membership values
%--------------------------------------------------------------------
%   PARAMETERS are to be specified accordingly for usage :
%          
%       Width of support if 'triangular'. Suggested is 5 for uint8
%       Width of support and spread factor if 'gaussian'. Suggested is
%       [5,2] for uint8
%       User defined membership values if 'custom'. Suggested is [1 2 3 4 5 4 3 2 1]
%-----------------------------------------------------------------------
close all;

ff = [9  ];
img = (imread('butterfly_noisy.bmp'));
imshow(fcnBPDFHE(img,  'triangular',ff) )
title('BPDFHE image - triangular ');

