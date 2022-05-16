
#ifndef camFusion_hpp
#define camFusion_hpp

#include "dataStructures.h"
#include "mykdtree.h"
#include <opencv2/core.hpp>
#include <stdio.h>
#include <vector>

void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT);
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches);
void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame);

void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait=true);

void clusterHelper(int index, std::vector<std::vector<float>>& points, std::vector<int>&  cluster, std::vector<bool>& processed, KdTree* tree, float distanceTol);
std::vector<std::vector<int>> myEuclideanCluster(std::vector<std::vector<float>>& points, KdTree* tree, float distanceTol);
std::vector<LidarPoint> removeLidarOutlier(const std::vector<LidarPoint> &lidarPoints, float clusterTolerance);

void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr,
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg=nullptr);
void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC);                  
#endif /* camFusion_hpp */