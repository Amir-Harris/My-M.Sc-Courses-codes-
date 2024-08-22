clc;
clear;
close all;

% random signal
x = randi(250,1,100);

%% spectrogram

mySpectrogram(x,40,true);

% scaleogram

%myScaleogram(x,[0.4,0.7,1,1.2,1.8,2.5,3.2,3,8,4],true);

myScaleogram(x,[0.5,0.2,1.2,1.5,1.8,2,3.5,3.8,4.5],true);