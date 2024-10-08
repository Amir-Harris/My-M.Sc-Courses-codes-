function H = hough_circles_acc(BW, radius)
    H = zeros(size(BW));
    for x = 1 : size(BW, 2)
        for y = 1 : size(BW, 1)
            if (BW(y,x))
                for theta = linspace(0, 2 * pi, 360)
                    a = round(x + radius * cos(theta));                
                    b = round(y + radius * sin(theta));
                    if (a > 0 && a <= size(H, 2) && b > 0 && b <= size(H,1))
                        H(b,a) = H(b,a) + 1;
                    end
                end
            end
        end
    end
end
