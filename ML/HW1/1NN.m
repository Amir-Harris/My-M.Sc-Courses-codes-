function [res] =  ocrClassify_1NN(trainingSet , feature)

% nearest neighbor algorithm
neighborDist = zeros(max([trainingSet.setCount]) , size(trainingSet,1));
neighborDist(:) =NaN;
for i=1:size(trainingSet,1)
    fSet = trainingSet(i);
    fSet = fSet.featureSet;
    
    for j=1:size(fSet,1)
        
        fMat = fSet{j,1};
        fDis = sum(sum(abs(feature - fMat)));
        neighborDist(j,i) = fDis;
    end
    
end
q = min(neighborDist);
[~ , classIndex]= min(q);
res = classIndex - 1;
end