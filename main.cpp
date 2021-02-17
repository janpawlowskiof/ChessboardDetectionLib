#include <iostream>
#include <chrono>
#include <opencv4/opencv2/core.hpp>

#include "chessboard_detection.h"


void process_camera_feed()
{
    auto cap = cv::VideoCapture(0);
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
        try{
            std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

            cv::Rect roi((frame_width-frame_size)/2,(frame_height-frame_size)/2,frame_size,frame_size);
            img = img(roi);
            cv::resize(img, img, cv::Size(512, 512));
            img = process_img(img);

            std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
            float fps = 1000000.0f / std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();

            std::string median_text = "Median: " + std::to_string(DEBUG_MEDIAN);
            cv::putText(img, median_text, cv::Point(30, 60), cv::FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2);

            std::string fps_text = std::to_string(fps) + " fps";
            cv::putText(img, fps_text, cv::Point(30, 30), cv::FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2);
        }
        catch (std::runtime_error error)
        {
            std::cout << error.what();
        }

        cv::imshow("camera feed", img);
        cv::waitKey(1);
    }

    cap.release();
    cv::destroyAllWindows();
}

void process_sample_image(std::string filename)
{
    auto img = cv::imread(filename);

    img = process_img(img);

    cv::imwrite("output.jpg", img);
    cv::imshow("camra feed", img);
    cv::waitKey(0);

    cv::destroyAllWindows();
}


int main() {
    std::cout << "Hello, World!" << std::endl;

//    process_camera_feed();
    process_sample_image("test7.jpg");

    return 0;
}
