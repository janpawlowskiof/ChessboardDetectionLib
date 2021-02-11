#include <opencv2/imgproc.hpp>

//
// Created by eg4l on 10.02.2021.
//
#pragma once


int median(cv::Mat& input)
{
    int rows = input.rows;
    int cols = input.cols;
    float histogram[256] = { 0 };
    for (int i = 0; i < rows; ++i)
    {
        ///Get the pointer of the first pixel of the i line
        const uchar *p = input.ptr<uchar>(i);
        ///Traverse the pixels in row i
        for (int j = 0; j < cols; ++j)
        {
            histogram[int(*p++)]++;
        }
    }
    int HalfNum = rows * cols / 2;
    int tempSum = 0;
    for (int i = 0; i < 255; i++)
    {
        tempSum = tempSum + histogram[i];
        if (tempSum > HalfNum)
        {
            return i;
        }
    }
    return 0;
}

cv::Mat auto_canny(cv::Mat img, float sigma)
{
    auto v = median(img);

    int lower = std::max(0.0, (1.0 - sigma) * v);
    int upper = std::min(255.0, (1.0 + sigma) * v);

    cv::Mat edges;
    cv::Canny(img, edges, lower, upper);

    cv::dilate(edges, edges, cv::Mat(), cv::Point(-1, -1), 2);
    cv::erode(edges, edges, cv::Mat(), cv::Point(-1, -1), 2);

    return edges;
}

cv::Mat simplify_image(cv::Mat img, float limit, cv::Size grid, int iters)
{
    cv::cvtColor(img, img, cv::COLOR_RGB2GRAY);
    cv::createCLAHE(limit, grid)->apply(img, img);

    if (limit){
        cv::morphologyEx(img, img, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(10, 10)));
    }

    return img;
}

void find_lines(cv::Mat edges, std::vector<cv::Vec2f>& lines)
{
    cv::HoughLines(edges, lines, 1, M_PI/180.0f, 30);
    lines.resize(200);
}

float normalize_angle(float angle)
{
    return std::fmod(std::fmod(angle, M_PI) + M_PI, M_PI);
}

bool is_horizontal(cv::Vec2f& line, float threshold)
{
    float theta = normalize_angle(line[1]);
    return M_PI/2 - threshold <= theta && theta <= M_PI/2 + threshold;
}

bool is_vertical(cv::Vec2f& line, float threshold)
{
    float theta = normalize_angle(line[1]);
    return (0 <= theta && theta <= threshold) || (M_PI - threshold <= theta && theta <= M_PI);
}

void split_lines_into_hv(std::vector<cv::Vec2f>& lines, std::vector<cv::Vec2f>& h_lines, std::vector<cv::Vec2f>& v_lines)
{
    h_lines.reserve(50);
    v_lines.reserve(50);

    for(auto& line : lines)
    {
        if(is_horizontal(line, 15.0f*M_PI/180.0f))
        {
            if(h_lines.size() < 50)
                h_lines.push_back(line);
        }
        else if(is_vertical(line, 15.0f*M_PI/180.0f))
        {
            if(v_lines.size() < 50)
                v_lines.push_back(line);
        }
    }
}

void overlay_lines(cv::Mat& img, std::vector<cv::Vec2f>& lines, cv::Scalar color)
{
    for(const auto& line : lines)
    {
        auto[rho, theta] = line.val;
        auto a = std::cos(theta);
        auto b = std::sin(theta);
        auto x0 = a*rho;
        auto y0 = b*rho;
        int x1 = (int)(x0 - 1000*b);
        int y1 = (int)(y0 + 1000*a);
        int x2 = (int)(x0 + 1000*b);
        int y2 = (int)(y0 - 1000*a);

        cv::line(img, cv::Point(x1, y1), cv::Point(x2, y2), color, 1);
    }
}


cv::Mat process_img(cv::Mat img)
{
    cv::resize(img, img, cv::Size(512, 512));
    auto simplified_image = simplify_image(img, 3, cv::Size(2, 6), 5);

    auto edges = auto_canny(simplified_image, 0.33f);

    std::vector<cv::Vec2f> lines;
    find_lines(edges, lines);

    std::vector<cv::Vec2f> h_lines;
    std::vector<cv::Vec2f> v_lines;
    split_lines_into_hv(lines, h_lines, v_lines);

    overlay_lines(img, v_lines, cv::Scalar(0, 255, 0));
    overlay_lines(img, h_lines, cv::Scalar(0, 0, 255));

    return img;
}
