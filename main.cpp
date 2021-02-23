#include <iostream>
#include <chrono>
#include <filesystem>
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

            cv::Mat simplified_image, intersections_mat, mozaic;
            IntersectionsVec intersections;
            process_img(img, intersections, intersections_mat, simplified_image, mozaic);

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


void display_sample_image(std::string input_path)
{
    auto img = cv::imread(input_path);

    int frame_width = img.cols;
    int frame_height = img.rows;
    int frame_size = std::min(frame_width, frame_height);

    cv::Rect roi((frame_width-frame_size)/2,(frame_height-frame_size)/2,frame_size,frame_size);
    img = img(roi);
    cv::resize(img, img, cv::Size(512, 512));

    cv::Mat simplified_image, intersections_mat, mozaic;
    IntersectionsVec intersections;

    process_img(img, intersections, intersections_mat, simplified_image, mozaic);

    cv::imshow("_mozaic.jpg", mozaic);
    cv::waitKey();
}


void split_image(std::string input_path, std::string output_root)
{
    std::string base_filename = input_path.substr(input_path.find_last_of("/\\") + 1);
    std::string::size_type const p(base_filename.find_last_of('.'));
    std::string file_without_extension = base_filename.substr(0, p);
    std::string output_path = output_root + "/" + file_without_extension + "/";
    output_path.erase(std::remove(output_path.begin(), output_path.end(), '.'), output_path.end());

    auto path = std::filesystem::path(output_path);
    std::filesystem::create_directories(path);

    auto img = cv::imread(input_path);

    int frame_width = img.cols;
    int frame_height = img.rows;
    int frame_size = std::min(frame_width, frame_height);

    cv::Rect roi((frame_width-frame_size)/2,(frame_height-frame_size)/2,frame_size,frame_size);
    img = img(roi);
    cv::resize(img, img, cv::Size(512, 512));
    cv::imwrite(output_path + "_original.jpg", img);

    cv::Mat simplified_image, intersections_mat, mozaic;
    IntersectionsVec intersections;

    process_img(img, intersections, intersections_mat, simplified_image, mozaic);
    split_image_into_folder(img, intersections, output_path);

    cv::imwrite(output_path + "_mozaic.jpg", mozaic);
    cv::imwrite(output_path + "_intersections.jpg", intersections_mat);
}

void test_creating_line(float v1, float v2, bool are_vertical)
{
    std::vector<cv::Vec2f> lines;
    if(are_vertical)
        lines.push_back(create_vertical_line(v1, v2, -512));
    else
        lines.push_back(create_horizontal_line(v1, 512, v2));

    auto line_wrappers = wrap_lines(lines);
    line_wrappers = recalculate_wrappers_properties(line_wrappers, are_vertical);
    assert(round(line_wrappers[0].position_at_min) == v1);
    assert(round(line_wrappers[0].position_at_max) == v2);
}

void test_creating_lines()
{
    test_creating_line(100, 200, true);
    test_creating_line(200, 200, true);
    test_creating_line(300, 200, true);

    test_creating_line(100, 200, false);
    test_creating_line(200, 200, false);
    test_creating_line(300, 200, false);
}

int main(int argc, const char *argv[]) {

//    if (argc < 3)
//    {
//        std::cout << "Specify paths!\n";
//        return -1;
//    }
//    std::string input_path = argv[1];
//    std::string output_path = argv[2];

//    test_creating_lines();
    display_sample_image("_original.jpg");

//    for(auto& p: std::filesystem::directory_iterator("/home/eg4l/Downloads/test3"))
//    {
//        split_image(p.path(), "/home/eg4l/Downloads/test4");
//    }

    return 0;
}
