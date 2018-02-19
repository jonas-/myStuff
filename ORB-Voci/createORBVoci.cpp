/**
 * ***********************************************************
 * Date:    December 2015
 * Author:  Jonas Eichenberger
 * License: GNU General Public License, GPL-3
 * Descrip: Create ORB vocabulary file from video ROS-bagfile
 * ***********************************************************
 */

#include <iostream>
#include <vector>

// DBoW2
#include <DBoW2ori/DBoW2.h>
#include <DUtils/DUtils.h>
#include <DUtilsCV/DUtilsCV.h> // defines macros CVXX
#include <DVision/DVision.h>

// OpenCV
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/nonfree/features2d.hpp>

// ROS
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <boost/foreach.hpp>
#include <cv_bridge/cv_bridge.h>

#include "ORBextractor.h"

using namespace DBoW2;
using namespace DUtils;
using namespace std;
using namespace ORB_SLAM;

// - - - - - --- - - - -- - - - - -

/// ORB Vocabulary
typedef DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB>
  ORBVocabulary;

/// ORB Database
typedef DBoW2::TemplatedDatabase<DBoW2::FORB::TDescriptor, DBoW2::FORB>
  ORBDatabase;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

void extractORBFeatures(cv::Mat &image, vector<vector<cv::Mat> > &features, ORBextractor* extractor);
void changeStructureORB( const cv::Mat &descriptor,vector<bool> &mask, vector<cv::Mat> &out);
void isInImage(vector<cv::KeyPoint> &keys, float &cx, float &cy, float &rMin, float &rMax, vector<bool> &mask);
void createVocabularyFile(ORBVocabulary &voc, std::string &fileName, const vector<vector<cv::Mat> > &features);

// ----------------------------------------------------------------------------

int main()
{
    // Extracting ORB features from bag file
    // = = =

    // load bag file
    string bagFile = "/path/to/your/bag/file";
    rosbag::Bag bag(bagFile);
    rosbag::View viewTopic(bag, rosbag::TopicQuery("/camera/image_raw"));
    int nImages = viewTopic.size();

    // initialze ORBextractor
    int nLevels = 6;
    ORBextractor* extractor = new ORBextractor(1000,1.2,nLevels,1,20);

    vector<vector<cv::Mat > > features;
    features.clear();
    features.reserve(nImages);

    cv_bridge::CvImageConstPtr cv_ptr;
    cv::Mat image;

    cout << "> Using bag file: " << bagFile << endl;
    cout << "> Extracting Features from " << nImages << " images..." << endl;
    BOOST_FOREACH(rosbag::MessageInstance const m, viewTopic)
    {
        sensor_msgs::Image::ConstPtr i = m.instantiate<sensor_msgs::Image>();

        if (i != NULL) {
            cv_ptr = cv_bridge::toCvShare(i);
            cvtColor(cv_ptr->image, image, CV_RGB2GRAY);

            extractORBFeatures(image, features, extractor);
        }

    }

    bag.close();

    cout << "... Extraction done!" << endl;


    // Creating the Vocabulary
    // = = =

    // define vocabulary
    const int k = 10; // branching factor
    const WeightingType weight = TF_IDF;
    const ScoringType score = L1_NORM;
    ORBVocabulary voc(k, nLevels, weight, score);

    std::string vociName = "vociOmni.txt";

    createVocabularyFile(voc, vociName, features);

    cout << "--- THE END ---" << endl;

    // = = =

    return 0;
}

// ----------------------------------------------------------------------------

void extractORBFeatures(cv::Mat &image, vector<vector<cv::Mat> > &features, ORBextractor* extractor) {
    vector<cv::KeyPoint> keypoints;
    cv::Mat descriptorORB;

    // extract
    (*extractor)(image, cv::Mat(), keypoints, descriptorORB);

    // reject features outside region of interest
    vector<bool> mask;
    float cx = 318.311759; float cy = 243.199269;
    float rMin = 50; float rMax = 240;
    isInImage(keypoints, cx, cy, rMin, rMax, mask);

    // create descriptor vector for the vocabulary
    features.push_back(vector<cv::Mat>());
    changeStructureORB(descriptorORB, mask, features.back());
}

// ----------------------------------------------------------------------------

void changeStructureORB( const cv::Mat &descriptor,vector<bool> &mask, vector<cv::Mat> &out) {
    for (int i = 0; i < descriptor.rows; i++) {
        if(mask[i]) {
            out.push_back(descriptor.row(i));
        }
    }
}

// ----------------------------------------------------------------------------

void isInImage(vector<cv::KeyPoint> &keys, float &cx, float &cy, float &rMin, float &rMax, vector<bool> &mask) {
    int N = keys.size();
    mask = vector<bool>(N, false);

    for(int i=0; i<N; i++) {
        cv::KeyPoint kp = keys[i];
        float uc = (kp.pt.x-cx);
        float vc = (kp.pt.y-cy);
        float rho = sqrt(uc*uc+vc*vc);

        if(rho>=rMin && rho<=rMax) {
            mask[i] = true;
        }
    }

}

// ----------------------------------------------------------------------------

void createVocabularyFile(ORBVocabulary &voc, std::string &fileName, const vector<vector<cv::Mat> > &features)
{

  cout << "> Creating vocabulary. May take some time ..." << endl;
  voc.create(features);
  cout << "... done!" << endl;

  cout << "> Vocabulary information: " << endl
  << voc << endl << endl;

  // save the vocabulary to disk
  cout << endl << "> Saving vocabulary..." << endl;
  voc.saveToTextFile(fileName);
  cout << "... saved to file: " << fileName << endl;
}

// ----------------------------------------------------------------------------


