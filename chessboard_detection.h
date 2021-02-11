//
// Created by eg4l on 10.02.2021.
//
#pragma once

#include <math.h>

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
//#include <opencv2/gapi/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/core/types.hpp>


int median(cv::Mat& input);
cv::Mat auto_canny(cv::Mat img, float sigma);
cv::Mat simplify_image(cv::Mat img, float limit, cv::Size grid, int iters);
void find_lines(cv::Mat edges, std::vector<cv::Vec2f>& lines);
float normalize_angle(float angle);
bool is_horizontal(cv::Vec2f& line, float threshold);
bool is_vertical(cv::Vec2f& line, float threshold);
void split_lines_into_hv(std::vector<cv::Vec2f>& lines, std::vector<cv::Vec2f>& h_lines, std::vector<cv::Vec2f>& v_lines);
void overlay_lines(cv::Mat& img, std::vector<cv::Vec2f>& lines, cv::Scalar color);
cv::Mat process_img(cv::Mat img);