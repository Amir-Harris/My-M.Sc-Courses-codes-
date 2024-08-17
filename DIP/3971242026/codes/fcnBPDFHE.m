function [final, transMap] = fcnBPDFHE(img, fuzzyType, param)

iptcheckinput(img,{'uint8','uint16','int16','single','double'}, {'nonsparse','2d'}, mfilename,'I',1);

if nargin == 1
    param = 5;
    memb = param(1)-abs(-param(1):param(1));
elseif nargin == 3
    if strcmp(fuzzyType,'triangular')
        if ~(numel(param)==1)
            error(' supports only 1 parameter for Triangular ');
        end
        memb = param(1)-abs(-param(1):param(1));
    elseif strcmp(fuzzyType,'gaussian')
        if ~(numel(param)==2)
            error(' supports only 2 param for Gaussian ');
        end
        memb = exp(-(-param(1):param(1)).^2/param(2)^2);
    elseif strcmp(fuzzyType,'custom')
        if numel(param)==0
            error('fcnBPDFHE requires the 1-D array for Custom ');
        end
        memb = param;
    else
        error('Unsupported memb type declaration');
    end
else
    error('Unsupported calling of fcnBPDFHE');
end

imageType = class(img);

% Hist creation
if strcmp(class(img),'uint8')
    [crispHist,grayScales] = imhist(img);
elseif strcmp(class(img),'uint16')
    crispHist = zeros([2^16 1]);
    for counter = 1:numel(img)
        crispHist(img(counter)+1) = crispHist(img(counter)+1) + 1;
    end
    grayScales = 0:(2^16 - 1);
elseif strcmp(class(img),'int16')
    crispHist = zeros([2^16 1]);
    for counter = 1:numel(img)
        crispHist(img(counter)+32769) = crispHist(img(counter)+32769) + 1;
    end
    grayScales = -32768:32767;
elseif (strcmp(class(img),'double')||strcmp(class(img),'single'))
    maxGray = max(img(:));
    minGray = min(img(:));
    img = im2uint8(mat2gray(img));
    [crispHist,grayScales] = imhist(img);
end

img = double(img);

fuzzyHist = zeros(numel(crispHist)+numel(memb)-1,1);

for counter = 1:numel(memb)
    fuzzyHist = fuzzyHist + memb(counter)*[zeros(counter-1,1);crispHist;zeros(numel(memb)-counter,1)];
end

fuzzyHist = fuzzyHist(ceil(numel(memb)/2):end-floor(numel(memb)/2));

del1FuzzyHist = [0;(fuzzyHist(3:end)-fuzzyHist(1:end-2))/2;0];
del2FuzzyHist = [0;(del1FuzzyHist(3:end)-del1FuzzyHist(1:end-2))/2;0];

locationIndex = (2:numel(fuzzyHist)-1)'+1;

maxLocAmbiguous = locationIndex(((del1FuzzyHist(1:end-2).*del1FuzzyHist(3:end))<0) & (del2FuzzyHist(2:end-1)<0));

counter = 1;

maxLoc = 1;

while counter < numel(maxLocAmbiguous)
    if (maxLocAmbiguous(counter)==(maxLocAmbiguous(counter+1)-1))
        maxLoc = [maxLoc ; (maxLocAmbiguous(counter)*(fuzzyHist(maxLocAmbiguous(counter))>fuzzyHist(maxLocAmbiguous(counter+1)))) + (maxLocAmbiguous(counter+1)*(fuzzyHist(maxLocAmbiguous(counter))<=fuzzyHist(maxLocAmbiguous(counter+1))))];
        counter = counter + 2;
    else
        maxLoc = [maxLoc ; maxLocAmbiguous(counter)];
        counter = counter + 1;
    end
end
if(maxLoc(end)~=numel(fuzzyHist))
    maxLoc = [maxLoc ; numel(fuzzyHist)];
end

low = maxLoc(1:end-1);
high = [maxLoc(2:end-1)-1;maxLoc(end)];
span = high-low;
cumulativeHist = cumsum(fuzzyHist);
M = cumulativeHist(high)-cumulativeHist(low);
factor = span .* log10(M);
range = max(grayScales)*factor/sum(factor);

transMap = zeros(numel(grayScales),1);

for counter = 1:length(low)
    for index = low(counter):high(counter)
        transMap(index) = round((low(counter)-1) + (range(counter)*(sum(fuzzyHist(low(counter):index)))/(sum(fuzzyHist(low(counter):high(counter))))));
    end
end

final = transMap(img+1);

final = mean(img(:))/mean(final(:))*final;

final = cast(final,imageType);

if strcmp(imageType,'single')
    final = minGray + (maxGray-minGray)*mat2gray(final);
elseif strcmp(imageType,'double')
    final = minGray + (maxGray-minGray)*mat2gray(final);
end