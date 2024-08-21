function [H, theta, rho] = hough_acc(BW, varargin)
   

    p = inputParser();
    addParameter(p, 'RhoResolution', 1);
    addParameter(p, 'Theta', linspace(-90, 89, 180));
    parse(p, varargin{:});

    rhoStep = p.Results.RhoResolution;
    theta = p.Results.Theta;


    dMax = sqrt((size(BW,1) - 1) ^ 2 + (size(BW,2) - 1) ^ 2);
    numRho = 2 * (ceil(dMax / rhoStep)) + 1;
    diagonal = rhoStep * ceil(dMax / rhoStep);
    numTheta = length(theta);
    H = zeros(numRho, numTheta);
    rho = -diagonal : diagonal;
    for i = 1 : size(BW,1)
        for j = 1 : size(BW,2)
            if (BW(i, j))
                for k = 1 : numTheta
                    temp = j * cos(theta(k) * pi / 180) + i * sin(theta(k) * pi / 180);
                    rowIndex = round((temp + diagonal) / rhoStep) + 1;
                    H(rowIndex, k) = H(rowIndex, k) + 1;                   
                end
            end            
        end
    end    
end
