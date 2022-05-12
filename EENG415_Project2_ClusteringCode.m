%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Clayton Daly, Jesse Dugan, Holly Hammons, Luke Logan
% EENG 415
% Dr. Salman Mohagheghi 
% 5/6/2022
% Project 2: Clustering Code
% NOTE: you will need the file "EENG415_Project2_FinalData.csv"
% in the same directory to run this file
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% read in the data, normalize
dataIn = readtable('EENG415_Project2_FinalData.csv');
dataIn = table2array(dataIn(:,2:end));
[rows, cols] = size(dataIn);
finalData = zeros(rows, (cols + 1));
for i = 1:rows
    finalData(i, 1) = i;
end
finalData(:, 2:end) = dataIn;
testData = normalize(finalData(:, 2:end));
%% Final Model Chosen: K-medoids
numClust = 4;
[kMedEuc, C,sumd,D] = kmedoids(testData, numClust, 'Distance', 'cosine');

% plot silhouette for final model
figure(1)
silhouette(testData,kMedEuc,'correlation');
title({'k-medoids correlation distance', 'k = 4'});

%% Intra-Cluster to Inter-Cluster Distance Ratio for final model
% note: the code for the distance ration was tested for a lot of different
% models, but is only written here once for documentation

% count the number of data instances in each cluster
C1 = 0;
C2 = 0;
C3 = 0;
C4 = 0;
distanceMet = 'correlation';
for i = 1:rows
    if kMedEuc(i) == 1
        C1 = C1 + 1;
    elseif kMedEuc(i) == 2
        C2 = C2 + 1;
    elseif kMedEuc(i) == 3
        C3 = C3 + 1;
    else
        C4 = C4 + 1;
    end
end

% collect all data points that are in each cluster into their own arrays
tempArray1 = [];
tempArray2 = [];
tempArray3 = [];
tempArray4 = [];
count = 1;
for i = 1:rows
    if kMedEuc(i) == 1
        tempArray1(count, :) = testData(i, :);
        count = count + 1;
    end
end
count = 1;
for i = 1:rows
    if kMedEuc(i) == 2
        tempArray2(count, :) = testData(i, :);
        count = count + 1;
    end
end
count = 1;
for i = 1:rows
    if kMedEuc(i) == 3
        tempArray3(count, :) = testData(i, :);
        count = count + 1;
    end
end
count = 1;
for i = 1:rows
    if kMedEuc(i) == 4
        tempArray4(count, :) = testData(i, :);
        count = count + 1;
    end
end

% calculate the intra cluster distance (numerator)
num1 = (sum(pdist(tempArray1,distanceMet))) / C1;
num2 = (sum(pdist(tempArray2,distanceMet))) / C2;
num3 = (sum(pdist(tempArray3,distanceMet))) / C3;
num4 = (sum(pdist(tempArray4,distanceMet))) / C4;

% find the distance from each point in one cluster to all of the points
% in the other clusters
% cluster 1
tempDist = cat(1,tempArray2, tempArray3, tempArray4);
sumDen1 = 0;
for i = 1:rows
    if kMedEuc(i) == 1
        newTempDist = cat(1, tempDist, testData(i, :));
        calcDist = squareform(pdist(newTempDist,distanceMet));
        sumDen1 = sumDen1 + sum(calcDist(:, end));
    end
end
den1 = sumDen1 / (C2 + C3 + C4);
% cluster 2
tempDist = cat(1,tempArray1, tempArray3, tempArray4);
sumDen2 = 0;
for i = 1:rows
    if kMedEuc(i) == 2
        newTempDist = cat(1, tempDist, testData(i, :));
        calcDist = squareform(pdist(newTempDist,distanceMet));
        sumDen2 = sumDen2 + sum(calcDist(:, end));
    end
end
den2 = sumDen2 / (C1 + C3 + C4);
% cluster 3
tempDist = cat(1,tempArray1, tempArray2, tempArray4);
sumDen3 = 0;
for i = 1:rows
    if kMedEuc(i) == 3
        newTempDist = cat(1, tempDist, testData(i, :));
        calcDist = squareform(pdist(newTempDist,distanceMet));
        sumDen3 = sumDen3 + sum(calcDist(:, end));
    end
end
den3 = sumDen3 / (C1 + C2 + C4);
% cluster 4
tempDist = cat(1,tempArray1, tempArray2, tempArray3);
sumDen4 = 0;
for i = 1:rows
    if kMedEuc(i) == 4
        newTempDist = cat(1, tempDist, testData(i, :));
        calcDist = squareform(pdist(newTempDist,distanceMet));
        sumDen4 = sumDen4 + sum(calcDist(:, end));
    end
end
den4 = sumDen4 / (C1 + C2 + C3);

% calculate the intra-cluster to inter-cluster distance ratio for all
% clusters
intraInter1 = num1 / den1
intraInter2 = num2 / den2
intraInter3 = num3 / den3
intraInter4 = num4 / den4
%% hierarchical (Holly)
% Note: The rest of the code in this file contains all of the models that 
% were not chosen. The remainder of the code consists of many different 
% clustering models and validation metrics.

% single linkage- cityblock
singLink = linkage(testData, 'single', 'cityblock');
figure(3)
dendrogram(singLink)
title('single linkage cityblock distance, k = 4');
c = cluster(singLink,'MaxClust',4); 
figure(4)
silhouette(testData,c, 'cityblock')
title('single linkage cityblock distance, k = 4');

% complete linkage- cityblock
singLink = linkage(testData, 'complete', 'cityblock');
figure(5)
dendrogram(singLink)
title('complete linkage cityblock distance, k = 4');
c = cluster(singLink,'MaxClust',4); 
figure(6)
silhouette(testData,c, 'cityblock')
title('complete linkage cityblock distance, k = 4');

% centroid linkage- cityblock
singLink = linkage(testData, 'centroid', 'cityblock');
figure(7)
dendrogram(singLink)
title('centroid linkage cityblock distance, k = 4');
c = cluster(singLink,'MaxClust',4); 
figure(8)
silhouette(testData,c, 'cityblock')
title('centroid linkage cityblock distance, k = 4');

% single linkage- euclidian
singLink = linkage(testData, 'single');
figure(9)
dendrogram(singLink)
title('single linkage euclidian distance, k = 4');
c = cluster(singLink,'MaxClust',4); 
figure(10)
silhouette(testData,c)
title('single linkage euclidian distance, k = 4');

% complete linkage- euclidian
singLink = linkage(testData, 'complete');
figure(11)
dendrogram(singLink)
title('complete linkage euclian distance, k = 4');
c = cluster(singLink,'MaxClust',4); 
figure(12)
silhouette(testData,c)
title('complete linkage euclian distance, k = 4');

% centroid linkage- euclidian
singLink = linkage(testData, 'centroid');
figure(13)
dendrogram(singLink)
title('centroid linkage euclidian distance, k = 4');
c = cluster(singLink,'MaxClust',4); 
figure(14)
silhouette(testData,c)
title('centroid linkage euclidian distance, k = 4');

% single linkage- chebychev
singLink = linkage(testData, 'single', 'chebychev');
figure(15)
dendrogram(singLink)
c = cluster(singLink,'MaxClust',4); 
figure(16)
silhouette(testData,c)
title('single linkage chebychev distance');

% complete linkage- chebychev
singLink = linkage(testData, 'complete', 'chebychev');
figure(17)
dendrogram(singLink)
c = cluster(singLink,'MaxClust',4); 
figure(18)
silhouette(testData,c)
title('complete linkage chebychev distance');

% centroid linkage- chebychev
singLink = linkage(testData, 'centroid', 'chebychev');
figure(19)
dendrogram(singLink)
c = cluster(singLink,'MaxClust',4); 
figure(20)
silhouette(testData,c)
title('centroid linkage chebychev distance');

% single linkage- mahalanobis
singLink = linkage(testData, 'single', 'mahalanobis');
figure(15)
dendrogram(singLink)
c = cluster(singLink,'MaxClust',3); 
figure(16)
silhouette(testData,c)
title('single linkage mahalanobis distance');

% complete linkage- mahalanobis
singLink = linkage(testData, 'complete', 'cosine');
figure(17)
dendrogram(singLink)
title('complete linkage cosine distance, k = 4');
c = cluster(singLink,'MaxClust',4); 
figure(18)
silhouette(testData,c)
title('complete linkage cosine distance, k = 4');

% centroid linkage- mahalanobis
singLink = linkage(testData, 'centroid', 'cosine');
figure(19)
dendrogram(singLink)
title('centroid linkage cosine distance, k = 4');
c = cluster(singLink,'MaxClust',4); 
figure(20)
silhouette(testData,c)
title('centroid linkage cosine distance, k = 4');

% % average linkage- mahalanobis
% singLink = linkage(testData, 'average', 'mahalanobis');
% figure(21)
% dendrogram(singLink)
% c = cluster(singLink,'MaxClust',4); 
% figure(22)
% silhouette(testData,c)
% title('average linkage mahalanobis distance, k = 4');
%% kmedoids (Holly)
% squared euclian
numClust = 4;
kMedEuc = kmedoids(testData, numClust, 'Distance', 'correlation');
figure(23)
silhouette(testData,kMedEuc, 'correlation')
title({'k-medoids squared cosine distance', 'k = 4'});
% mahalanobis
% kMedMah = kmedoids(finalData, numClust, 'Distance', 'mahalanobis');
% figure(24)
% silhouette(finalData,kMedMah)
% title('k-medoids mahalanobis distance');
% cityblock
kMedCity = kmedoids(finalData, numClust, 'Distance', 'cityblock');
figure(25)
silhouette(finalData,kMedCity)
title('k-medoids cityblock distance');
% euclidean
% kMedEuc = kmedoids(finalData, numClust, 'Distance', 'euclidean');
% figure(26)
% silhouette(finalData,kMedEuc)
% title('k-medoids euclidian distance');
% chebychev
kMedCheb = kmedoids(finalData, numClust, 'Distance', 'chebychev');
figure(27)
silhouette(finalData,kMedCheb)
title('k-medoids chebychev distance');
%% k-means and k-medoids clustering (Jesse)
% Use the elbow method to find the optimal number of clusters for different distance functions
% Square euclidean distance
values_sqeuclidean = [];
for k = 1:20
    [idx,C,sumd] = kmeans(testData,k,'Distance','sqeuclidean');
    avg_sumd = sum(sumd)/k;
    values_sqeuclidean(k) = avg_sumd;
end

% Cityblock (Manhattan distance)
values_cityblock = [];
for k = 1:20
    [idx,C,sumd] = kmeans(testData,k,'Distance','cityblock');
    avg_sumd = sum(sumd)/k;
    values_cityblock(k) = avg_sumd;
end

% Cosine distance
values_cosine = [];
for k = 1:20
    [idx,C,sumd] = kmeans(testData,k,'Distance','sqeuclidean');
    avg_sumd = sum(sumd)/k;
    values_cosine(k) = avg_sumd;
end

% Correlation distance
values_correlation = [];
for k = 1:20
    [idx,C,sumd] = kmeans(testData,k,'Distance','sqeuclidean');
    avg_sumd = sum(sumd)/k;
    values_correlation(k) = avg_sumd;
end

% Squared euclidean distance
kmedoids_values_sqeuclidean = [];
for k = 1:20
    [idx,C,sumd] = kmedoids(testData,k,'Distance','sqEuclidean');
    avg_sumd = sum(sumd)/k;
    kmedoids_values_sqeuclidean(k) = avg_sumd;
end

% Euclidean distance
values_euclidean = [];
for k = 1:20
    [idx,C,sumd] = kmedoids(testData,k,'Distance','euclidean');
    avg_sumd = sum(sumd)/k;
    values_euclidean(k) = avg_sumd;
end

% Standardized euclidean
values_seuclidean = [];
for k = 1:20
    [idx,C,sumd] = kmedoids(testData,k,'Distance','seuclidean');
    avg_sumd = sum(sumd)/k;
    values_seuclidean(k) = avg_sumd;
end

% Cityblock (Manhattan)
kmedoids_values_cityblock = [];
for k = 1:20
    [idx,C,sumd] = kmedoids(testData,k,'Distance','cityblock');
    avg_sumd = sum(sumd)/k;
    kmedoids_values_cityblock(k) = avg_sumd;
end

% Cosine
kmdeoids_values_cosine = [];
for k = 1:20
    [idx,C,sumd] = kmedoids(testData,k,'Distance','cosine');
    avg_sumd = sum(sumd)/k;
    kmedoids_values_cosine(k) = avg_sumd;
end

% Correlation
kmedoids_values_correlation = [];
for k = 1:20
    [idx,C,sumd] = kmedoids(testData,k,'Distance','correlation');
    avg_sumd = sum(sumd)/k;
    kmedoids_values_correlation(k) = avg_sumd;
end

figure
plot(values_sqeuclidean, 'LineWidth',1)
hold on
plot(values_cityblock, 'LineWidth',1)
hold on
plot(values_cosine, '--', 'LineWidth',1)
hold on
plot(values_correlation, ':', 'LineWidth',1)
hold on
plot(kmedoids_values_sqeuclidean, 'LineWidth',1)
hold on
plot(kmedoids_values_cityblock,'--', 'LineWidth',1)
hold on
plot(values_euclidean, 'LineWidth',1)
hold on
plot(kmedoids_values_cosine, '--', 'LineWidth',1)
hold on
plot(kmedoids_values_correlation, ':', 'LineWidth',1)

title('Optimal value of k')
xlabel('Number of clusters')
ylabel('Average within-cluster sum of point-to-centroid distances')
legend('k-means squared euclidean','k-means cityblock','k-means cosine','k-means correlation','k-medoids squared euclidean','k-medoids cityblock','k-medoids euclidean','k-medoids cosine','k-medoids correlation')

%% Plot silhouettes for the best performing k-means models (Jesse)
% note: since K-means begins with randomly assigning centroids, the results
% change with each time we run the code, some silhouette plots are therefore
% better than others even for the same parameters
clust1 = kmeans(testData,3,'Distance','cityblock');
clust2 = kmeans(testData,4,'Distance','cityblock');
clust3 = kmeans(testData,5,'Distance','cityblock');
clust4 = kmeans(testData,4,'Distance','sqEuclidean');

figure
silhouette(testData,clust1,'cityblock');
title('Silhouette plot for k-means with cityblock distance, k = 3')
ylabel('Cluster')
xlabel('Silhouette Value')

figure
silhouette(testData,clust2,'cityblock')
title('Silhouette plot for k-means with cityblock distance, k = 4')
ylabel('Cluster')
xlabel('Silhouette Value')

figure
silhouette(testData,clust3,'cityblock')
title('Silhouette plot for k-means with cityblock distance, k = 5')
ylabel('Cluster')
xlabel('Silhouette Value')

figure
silhouette(testData,clust4,'sqEuclidean')
title('Silhouette plot for k-means with squared euclidean distance, k = 4')
ylabel('Cluster')
xlabel('Silhouette Value')

% Best model for k-means overall is cityblock distance and k = 4, but it is
% not reliably good becuase of the algorithm's approach (begins with a
% random selection)

% Plot silhouette for the best performing k-medoids models

clust1_medoids = kmedoids(testData,3,'Distance','cosine');
clust2_medoids = kmedoids(testData,4,'Distance','cosine');
clust3_medoids = kmedoids(testData,5,'Distance','cosine');

clust4_medoids = kmedoids(testData,3,'Distance','correlation');
clust5_medoids = kmedoids(testData,4,'Distance','correlation');
clust6_medoids = kmedoids(testData,5,'Distance','correlation');

clust7_medoids = kmedoids(testData,3,'Distance','euclidean');
clust8_medoids = kmedoids(testData,4,'Distance','euclidean');
clust9_medoids = kmedoids(testData,5,'Distance','euclidean');

figure
silhouette(testData,clust1_medoids, 'cosine')
title('Silhouette plot for k-medoids with cosine distance, k = 3')
ylabel('Cluster')
xlabel('Silhouette Value')

figure
silhouette(testData,clust2_medoids, 'cosine')
title('Silhouette plot for k-medoids with cosine distance, k = 4')
ylabel('Cluster')
xlabel('Silhouette Value')

figure
silhouette(testData,clust3_medoids, 'cosine')
title('Silhouette plot for k-medoids with cosine distance, k = 5')
ylabel('Cluster')
xlabel('Silhouette Value')

figure
silhouette(testData,clust4_medoids, 'correlation')
title('Silhouette plot for k-medoids with correlation distance, k = 3')
ylabel('Cluster')
xlabel('Silhouette Value')

figure
silhouette(testData,clust5_medoids, 'correlation')
title('Silhouette plot for k-medoids with correlation distance, k = 4')
ylabel('Cluster')
xlabel('Silhouette Value')

figure
silhouette(testData,clust6_medoids, 'correlation')
title('Silhouette plot for k-medoids with correlation distance, k = 5')
ylabel('Cluster')
xlabel('Silhouette Value')

figure
silhouette(testData,clust7_medoids, 'euclidean')
title('Silhouette plot for k-medoids with euclidean distance, k = 3')
ylabel('Cluster')
xlabel('Silhouette Value')

figure
silhouette(testData,clust8_medoids, 'euclidean')
title('Silhouette plot for k-medoids with euclidean distance, k = 4')
ylabel('Cluster')
xlabel('Silhouette Value')

figure
silhouette(testData,clust9_medoids, 'euclidean')
title('Silhouette plot for k-medoids with euclidean distance, k = 5')
ylabel('Cluster')
xlabel('Silhouette Value')