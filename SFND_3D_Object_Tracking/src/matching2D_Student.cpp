
#include <numeric>
#include "matching2D.hpp"

using namespace std;
int TRESHOLD = 100;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;
    int k = 2; // k nearest neighbors

    

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = descriptorType.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if(descSource.type()!= CV_32F){
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }

    // perform matching task
    if (selectorType == "SEL_NN")
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType == "SEL_KNN")
    { // k nearest neighbors (k=2)
        vector<vector<cv::DMatch>> knnMatches;
        matcher->knnMatch(descSource, descRef, knnMatches, k);
        double minDistRatio = 0.8;
        
        for (auto& i:knnMatches){
            double distanceTemp = i[0].distance/i[1].distance; // d(fa,fb1)/d(fa,fb2)
            if(distanceTemp < minDistRatio){
                matches.push_back(i[0]);
            }
        }

    }

    cout << "number of matched keypoints: " << matches.size() << endl;
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType, float &time)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    int threshold = 30;        // FAST/AGAST detection threshold score.
    int octaves = 3;           // detection octaves (use 0 to do single scale)
    float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

    if (descriptorType =="BRISK")
    {
       
        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    // BRIEF, ORB, FREAK, AKAZE, SIFT
    
    else if (descriptorType == "BRIEF")
    {
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create(64);
    }
    else if (descriptorType =="ORB")
    {

        extractor = cv::ORB::create();
    }
    else if (descriptorType == "FREAK")
    {

        extractor = cv::xfeatures2d::FREAK::create();
    }
    else if (descriptorType == "SIFT")/// This cannot be used with binary matches
    {
        extractor = cv::xfeatures2d::SIFT::create();
    }
    else if (descriptorType == "AKAZE")
    {
        extractor = cv::AKAZE::create();
    
    } 
    

    // perform feature description
    double t = (double)cv::getTickCount();
    //if(descriptorType.compare("AKAZE") == 0)
    //    extractor->detectAndCompute(img, cv::noArray(), keypoints, descriptors);
    //else
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
    time = time + (1000 * t / 1.0);
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, float &time, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    time = time + (1000 * t / 1.0);

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Shi-Tomasi Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }
}

void detKeypointsHarris(vector<cv::KeyPoint> &keypoints, cv::Mat &img, float &time, bool bVis){
     // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    int apertureSize = 3;
    int treshold = TRESHOLD; //Try different values
    double k = 0.04;

    double t = (double)cv::getTickCount();

    cv::Mat dst_norm, dst_norm_scaled;
    cv::Mat dst = cv::Mat::zeros(img.size(), CV_32FC1);
    cornerHarris(img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
    normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
    convertScaleAbs(dst_norm, dst_norm_scaled);


    for(int i = 0; i<dst_norm.rows; i++){
        for(int j = 0; j < dst_norm.cols; j++){
            int response = (int) dst_norm.at<float>(i, j);
            if(response>treshold){
                cv::KeyPoint keyPointBest;
                keyPointBest.pt = cv::Point2f(j,i);
                keyPointBest.size = 2*apertureSize;
                keyPointBest.response = response;


                bool overlap = false;
                for(auto it = keypoints.begin(); it!= keypoints.end(); ++it){
                    double overlappedKey = cv::KeyPoint::overlap(keyPointBest, *it);
                    if(overlappedKey>0.0){ // we do not tolerate any overlap
                        overlap = true;
                        if(keyPointBest.response > (*it).response) {
                            *it = keyPointBest;
                            break;
                        }
                    }
                }
                if(!overlap){
                    keypoints.push_back(keyPointBest);
                }
            }
        }
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Harris detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    time = time + (1000 * t / 1.0);

    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Harris Corner Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }    

}
void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, float &time, bool bVis){

    double t = (double)cv::getTickCount();

    if(detectorType == "FAST"){
        cv::Ptr<cv::FastFeatureDetector> fastPtr = cv::FastFeatureDetector::create(TRESHOLD, true);
        fastPtr->detect(img, keypoints);
    }
    else if(detectorType == "BRISK"){
        int octaves = 4;
        int patternS = 0.01f;
        cv::Ptr<cv::FeatureDetector> briskPtr = cv::BRISK::create(TRESHOLD, octaves, patternS);
        briskPtr->detect(img,keypoints);
        
    }
    else if(detectorType == "ORB"){
        cv::Ptr<cv::ORB> orbPtr = cv::ORB::create();
        orbPtr->detect(img, keypoints);
    }
    else if(detectorType == "AKAZE"){
        cv::Ptr<cv::FeatureDetector> akazePtr = cv::AKAZE::create();
        akazePtr->detect(img, keypoints);
    }
    
    else if(detectorType == "SIFT"){ // Default is SIFT
        cv::Ptr<cv::xfeatures2d::SIFT> siftPtr = cv::xfeatures2d::SIFT::create();
        siftPtr->detect(img, keypoints);
    }

    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << detectorType << " detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    time = time + (1000 * t / 1.0);


    // visualize results
    if (bVis)
    {
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = "Detector Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }    
}
