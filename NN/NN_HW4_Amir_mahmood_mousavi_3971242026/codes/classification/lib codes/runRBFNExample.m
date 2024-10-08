clear;

% Remove any holds on the existing plots.
figure(1);
hold off;

figure(2);
hold off;

% Add the subdirectories to the path.
addpath('kMeans');
addpath('RBFN');

% Load the data set. 
% This loads two variables, X and y.
%   X - The dataset, 1 sample per row.
%   y - The corresponding label (category 1 or 2).
% The data is randomly sorted and grouped by category.
data = load('dataset.csv');
X = data(:, 1:2);
y = data(:, 3);
figure(3);
axis([-5 5 -5 5]);
% Plot the data set and neuron prototypes.
plot(X, 'ro');

% Set 'm' to the number of data points.
m = size(X, 1);

% ===================================
%     Train RBF Network
% ===================================

disp('Training the RBFN...');

% Train the RBFN using 10 centers per category.
[Centers, betas, Theta] = trainRBFN(X, y, 5, true);
 
% ================================
%         Contour Plots
% ================================

disp('Evaluating RBFN over input space...');

% Define a grid over which to evaluate the RBFN.
gridSize = 50;
u = linspace(-2, 2, gridSize);
v = linspace(-2, 2, gridSize);

% We'll store the scores for each category as well as the 'prediction' for
% each point on the grid.
scores1 = zeros(length(u), length(v));
scores2 = zeros(length(u), length(v));
scores3 = zeros(length(u), length(v));
p = zeros(length(u), length(v));

% Evaluate the RBFN over the grid.
% For each row of the grid...
for (i = 1 : length(u))
    
    % Report our progress every 10th row.
    if (mod(i, 10) == 0)
        fprintf('  Grid row = %d / %d...\n', i, gridSize);
        if exist('OCTAVE_VERSION') fflush(stdout); end;
    end
    
    % For each column of the grid...
    for (j = 1 : length(v))
        
        % Compute the category 1 and 2 scores.
		scores = evaluateRBFN(Centers, betas, Theta, [u(i), v(j)]);
        
		scores1(i, j) = scores(1, 1);
        scores2(i, j) = scores(2, 1);
%         scores3(i, j) = scores(3, 1);
        
        % Pick the higher score.
        if (scores1(i, j) == scores2(i, j))
            p(i,j) = 1.5;
        elseif (scores1(i, j) > scores2(i, j))
            p(i, j) = 1;
        else 
            p(i, j) = 2;
        end
    end
end

% Contour Plot #1: Plot the category 1 score.
% Plot the contour lines to show the continuous-valued function which is the
% output of the category 2 node RBFN.
figure(1);
[C, h] = contour(u, v, scores1');
%set(h,'ShowText','on','TextStep',get(h,'LevelStep')*2);
hold on;
axis([-5 5 -5 5]);
plot(Centers(:, 1), Centers(:, 2), 'k*');
plot(X(y == 2, 1), X(y == 2, 2), 'bx');
plot(X(y == 1, 1), X(y == 1, 2), 'go');

title('Category 1 Output');

fprintf('Minimum category 1 score: %.2f\n', min(min(scores1)));
fprintf('Maximum category 1 score: %.2f\n', max(max(scores1)));
if exist('OCTAVE_VERSION') fflush(stdout); end;

% Contour Plot #2: Plot the approximate decision boundary over the dataset.
figure(2);

% Plot the data set and neuron prototypes.
plot(X(y == 1, 1), X(y == 1, 2), 'ro');
hold on;
axis([-10 10 -10 10]);
plot(X(y == 2, 1), X(y == 2, 2), 'bx');
plot(X(y == 3, 1), X(y == 3, 2), 'g+');
plot(Centers(:, 1), Centers(:, 2), 'k*');

% Draw a contour plot line where p = 1.5. The contour function performs
% some interpolation of the points for you.
[c, h] = contour(u, v, p', [1.5, 1.5]);
title('Decision Boundary');

% ========================================
%       Measure Training Accuracy
% ========================================

disp('Measuring training accuracy...');

numRight = 0;

wrong = [];

% For each training sample...
for (i = 1 : m)
    % Compute the scores for both categories.
    scores = evaluateRBFN(Centers, betas, Theta, X(i, :));
    
	[maxScore, category] = max(scores);
	
    % Validate the result.
    if (category == y(i))
        numRight = numRight + 1;
    else
        wrong = [wrong; X(i, :)];
    end
    
end

% Mark the incorrectly recognized samples with a black asterisk.
%plot(wrong(:, 1), wrong(:, 2), 'k*');

accuracy = numRight / m * 100;
fprintf('Training accuracy: %d / %d, %.1f%%\n', numRight, m, accuracy);
if exist('OCTAVE_VERSION') fflush(stdout); end;
