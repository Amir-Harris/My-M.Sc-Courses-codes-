function [res] =  ocrClassify_KNN(trainingSet , feature , K)
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

classIndexes=[];

for i=1:K
    q = min(neighborDist);
    [~ , classIndex]= min(q);
    classIndexes = [classIndexes ; classIndex - 1];
    [~ , minIndex] = min(neighborDist(:,classIndex));
    neighborDist(minIndex,classIndex) = NaN;
end

res = mode(classIndexes);
end