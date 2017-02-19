function ber = BER(confMat)
% BER calculate the balanced error rate.
%input: confusion matrix (Dimensions N x N... 
%where N is the number of classes
%output: balanced error rate (scalar)
    ber = 0;
    rs = sum(confMat,2);
    for i = 1:size(confMat,1)
        ls = 0;
        for j = 1:size(confMat,2)
            if i == j
                continue
            end
            ls = ls + confMat(i,j);
        end
        ber = ber + ls/rs(i);
    end
    ber = ber / size(confMat,1);
end