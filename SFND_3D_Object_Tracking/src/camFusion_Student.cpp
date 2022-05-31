
#include <iostream>
#include <algorithm>
#include <numeric>
#include <math.h>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"

using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        // pixel coordinates
        pt.x = Y.at<double>(0, 0) / Y.at<double>(2, 0); 
        pt.y = Y.at<double>(1, 0) / Y.at<double>(2, 0); 

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}

/* 
* The show3DObjects() function below can handle different output image sizes, but the text output has been manually tuned to fit the 2000x2000 size. 
* However, you can make this function work for other sizes too.
* For instance, to use a 1000x1000 size, adjusting the text positions by dividing them by 2.
*/
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);  
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, cv::WINDOW_NORMAL);
    cv::imshow(windowName, topviewImg);

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    //Check if it is inside bounding box
    std::vector<cv::DMatch> roiBox;
    std::vector<float> distance;
    float distSum = 0;
    float mean;
    for(auto& match : kptMatches){
        auto &kptPrevPT = kptsPrev[match.queryIdx].pt;
        auto &kptCurrPT = kptsPrev[match.trainIdx].pt;

        if(boundingBox.roi.contains(kptPrevPT)){
            roiBox.push_back(match);
            float tempDist = cv::norm(kptCurrPT-kptPrevPT);
            distance.push_back(tempDist);
            distSum += tempDist;
        }
    }

    mean = distSum/distance.size();

    for(int i = 0; i< distance.size(); ++i){
        if(distance[i] < mean){
            boundingBox.keypoints.push_back(kptsCurr[roiBox[i].trainIdx]);
            boundingBox.kptMatches.push_back(roiBox[i]);
        }
    }
    
    
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    double dt = 1.0/frameRate;

    vector<double> distRatios;
    
    for(auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; it1++){
        //cout<< "Starting Loop: " << it1->trainIdx <<endl;
        //get current keypoint and its matched partner in the prev frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        //cout<< "First keypoints created" <<endl;
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);
       

        for(auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); it2++){
            //compute distance and distance ratios
            double minDist = 100.0;

            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);
            //cout<< "Second keypoints created" <<endl;

            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);


            if(distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist){
                double distRatio = distCurr / distPrev;
                distRatios.push_back(distRatio);
            }
        }
    }
    //cout<< "Finished iteration " <<endl;

    

    if(distRatios.size() == 0){
        TTC = NAN;
        return;
    }

    //computer camera based TTC from distance ratios
    std::sort(distRatios.begin(), distRatios.end());
    long medInd = floor(distRatios.size() / 2.0);
    double meanDistRatio = (distRatios.size() % 2 == 0) ? (distRatios[medInd-1] + distRatios[medInd])/2.0 : distRatios[medInd];


    TTC = -dt / (1-meanDistRatio);

    std::cout<< "TTC Camera: " << TTC << endl;





}




std::vector<LidarPoint> filterOutliers(const std::vector<LidarPoint> &lidarPoints){

    // Filtering by distance to the average of the cluster
    	std::vector<LidarPoint> filteredData;
	double pointsSum = 0;
	
	for(auto& it: lidarPoints){
		pointsSum = pointsSum + it.x;
	}
	double meanPoints = pointsSum/lidarPoints.size();
	
	double pointsVariance = 0;
	
	for (auto& it : lidarPoints){
		pointsVariance += pow((it.x - meanPoints),2);
	}
	pointsVariance = pointsVariance/(lidarPoints.size()-1);
	double stdValue = sqrt(pointsVariance);
	
	double upperLimit = meanPoints + 2*stdValue;
	double lowerLimit = meanPoints - 2*stdValue;
	
	for(int i = 0; i < lidarPoints.size() - 1; i++){
		if((lidarPoints[i].x <= upperLimit) && (lidarPoints[i] >= lowerLimit)){
			filteredData.push_back(lidarPoints[i]);
		}
	}
	
	return filteredData;

}

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev, std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    double dt = 1/frameRate;
    double lanewidth = 4.0; // ego line of 4 meters is assumed
    float clusterTolerance = 0.1;
    std::cout.flush();


    double minPrev = 0, minCurr = 0;
    vector<double> xPrev, xCurr;

    std::vector<LidarPoint> lidarPointsPrevClustered = filterOutliers(lidarPointsPrev);

    std::vector<LidarPoint> lidarPointsCurrClustered = filterOutliers(lidarPointsCurr);

    for (auto& it : lidarPointsPrevClustered){
        if(abs(it.y) <= lanewidth / 2.0){
            xPrev.push_back(it.x);
        }
    }

    for (auto& it : lidarPointsCurrClustered){
        if(abs(it.y) <= lanewidth / 2.0){
           yPrev.push_back(it.x);
        }
    }
	
    if(xCurr.size() >0){
	    for(auto x:xCurr){
		    minCurr+=x;
	    }
	    minCurr=minCurr/xCurr.size();
    }
    if(xPrev.size() >0){
	    for(auto x:xPrev){
		    minPrev+=x;
	    }
	    minPrev=minPrev/xPrev.size();
    }
    TTC = minCurr * dt / (minPrev- minCurr);
    cout << "MinXPrev: " << minPrev << " MinXCurr: "<<minCurr << endl;
    cout << "TTC from lidar is: " << TTC << endl;
}

void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    
    for(auto& prevBox: prevFrame.boundingBoxes){
        std::cout.flush();
        std::map<int, int> mat;  // Create the map with (boxID_prev, boxID_curr) we need to see which box has the most of this
        //prevBox.kptMatches = prevFrame.kptMatches;
        for(auto& currBox: currFrame.boundingBoxes){
            //currBox.kptMatches = currFrame.kptMatches;
            for(auto &match: matches){
                auto &prevKpt = prevFrame.keypoints[match.queryIdx].pt;
                auto &currkpt = currFrame.keypoints[match.trainIdx].pt;
                if(prevBox.roi.contains(prevKpt) && currBox.roi.contains(currkpt)){
                    if(mat.count(currBox.boxID)==0) // Verification to initialize that key value
                        mat[currBox.boxID]=1;
                    else
                        mat[currBox.boxID]++;                    
                
                }
            }
            
        }


        
        // Getting the maximum value
        
        int arg_max = 0;
        int currentMax = 0;
        for(auto it = mat.begin(); it!=mat.end(); ++it){
            if(it->second > currentMax){
                arg_max = it->first;
                currentMax = it->second;
            }
        }
        
        bbBestMatches[prevBox.boxID] = arg_max;

        
        //std::cout<<"ID Match: " << prevBox.boxID << " => " << arg_max << " Size: " << prevBox.kptMatches.size() << std::endl; 
    }
    
}
