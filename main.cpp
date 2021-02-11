#include <iostream>
#include <opencv4/opencv2/core.hpp>
#include <opencv4/opencv2/core.hpp>

#include "chessboard_detection.h"


void process_camera_feed()
{
    auto cap = cv::VideoCapture(1);
    if(!cap.isOpened())
    {
        std::cout << "Unable to process camera feed!\n";
        cap.release();
        return;
    }

    int frame_width = (int)cap.get(3);
    int frame_height = (int)cap.get(4);
    int frame_size = std::min(frame_width, frame_height);

    std::cout << "Camera feed size: " << frame_width << " x " << frame_height << "\n";
    cv::Mat img;

    while(cap.read(img))
    {
        img = process_img(img);

        cv::imshow("camra feed", img);
        cv::waitKey(1);
    }

    cap.release();
    cv::destroyAllWindows();
}

void process_sample_image()
{

    auto img = cv::imread("test.jpg");

    img = process_img(img);

    cv::imwrite("output.jpg", img);
    cv::imshow("camra feed", img);
    cv::waitKey(0);

    cv::destroyAllWindows();
}


int main() {
    std::cout << "Hello, World!" << std::endl;

//    process_camera_feed();
    process_sample_image();

    return 0;
}
