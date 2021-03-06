# SFND 3D Object Tracking

Welcome to the final project of the camera course. By completing all the lessons, you now have a solid understanding of keypoint detectors, descriptors, and methods to match them between successive images. Also, you know how to detect objects in an image using the YOLO deep-learning framework. And finally, you know how to associate regions in a camera image with Lidar points in 3D space. Let's take a look at our program schematic to see what we already have accomplished and what's still missing.

<img src="images/course_code_structure.png" width="779" height="414" />

In this final project, you will implement the missing parts in the schematic. To do this, you will complete four major tasks: 
1. First, you will develop a way to match 3D objects over time by using keypoint correspondences. 
2. Second, you will compute the TTC based on Lidar measurements. 
3. You will then proceed to do the same using the camera, which requires to first associate keypoint matches to regions of interest and then to compute the TTC based on those matches. 
4. And lastly, you will conduct various tests with the framework. Your goal is to identify the most suitable detector/descriptor combination for TTC estimation and also to search for problems that can lead to faulty measurements by the camera or Lidar sensor. In the last course of this Nanodegree, you will learn about the Kalman filter, which is a great way to combine the two independent TTC measurements into an improved version which is much more reliable than a single sensor alone can be. But before we think about such things, let us focus on your final project in the camera course. 

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level project directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./3D_object_tracking`.


## Procedure

1. Match 3D Objects

matchBoundingBoxes Function was implemented in camFusion_student.cpp file.
It consisted in 2 for loops dedicated to go over the current and previous box on the images and select the the highest values to keep track on.
<img src="images/matchBoundingBoxes.png" width="779" height="414" />

2. Lidar-Based TTC
It was required to compute the TTC equation: minCurr*DT / (minPrev-minCurr)

To to it, first the lidar points needed to be filtered, to avoid any noise. 
To remove outliers we filtered the Lidar data on X values based on standard deviation.
By identifying data that was out of the upper and lower limits, the data to calculate TTC was based only the filtered.
<img src="images/filteredDataLidar.png" width="779" height="414" />

Afterwards, the closer points from the lidar roi data were gathered (from the identified object) and the TTC was computed
<img src="images/lidarTTC.png" width="779" height="414" />


3. Camera-based TTC
Compute the TTC equation now for camera data.
It was required to find keypoint matches in the expected region of interest. 
Keypoints matches were done fairly easy, similar to the last project implementing SIFT, ORB, FREAK, AKAZE and BRIEF extractors.
With these values and storing the matches in a specific object (bbBestMatches).

The computation was done by going over each match and gathering the information between the previous and current camera data.
Formula in this case was -DT/ (1-meanDistRatio)



## Evaluation 1:
Find where the Lidar values show incorrect or way off values:

Figure

<img src="images/TTC_34.png" width="779" height="414" />
TTC Lidar: 34.3404 s
Distance is at 7.55 m, however, since the TTC calculation depends on the previous value, it will show as increasing the time since the delta will give a higher value.

<img src="images/TTC_3_83.png" width="779" height="414" />
TTC Lidar: 3.83 seconds
Distance is at 7.205. Same as before, Lidar values give a TTC based on the previous X value, which means it will be based on how faster the front car hits the break. 

<img src="images/TTC_negative.png" width="779" height="414" />
TTC Lidar shows negative?
Since previous data was lower than current, the front car accelerated, changing how the value is calculated.


As discussed on the previous images, the TTC calculation on the Lidar depends on the values coming from the previous data, and the keypoints detected, so when a car is disaccelerating, the equation will give that previous data was higher, giving incorrect, negative or even values higher than expected.
The current equation used is for constant velocity, so it cannot handle data with sudden speed changes






## Evaluation 2:

Showing the plots for all combinations:
<img src="images/shitomasiResults.png" width="779" height="414" />
<img src="images/HarrisResults.png" width="779" height="414" />
<img src="images/FastResults.png" width="779" height="414" />
<img src="images/BRISKResults.png" width="779" height="414" />
<img src="images/ORBResults.png" width="779" height="414" />
<img src="images/AKAZE Results.png" width="779" height="414" />
<img src="images/SIFTResults.png" width="779" height="414" />

From these graphs we can see that the two combinations with more stability on results were using shitomasi and AKAZE with any Extractor type.

Stability refering it to the ability for gathering more matches between images and calculating a logical TTC.
Non logical data was taken on values of more than 20 or negative values.


