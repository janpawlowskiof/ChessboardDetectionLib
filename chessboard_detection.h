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


struct LineWrapper
{
    cv::Vec2f value;
    float position_at_min = 0;
    float position_at_center = 0;
    float position_at_max = 0;
    float ratio_at_min = 0;
    float ratio_at_max = 0;
    float offset_from_prev = INFINITY;
    float offset_to_next = INFINITY;
};

extern float DEBUG_MEDIAN;

typedef std::vector<std::vector<cv::Vec2f>> IntersectionsVec;

int median(cv::Mat& input);
cv::Mat auto_canny(cv::Mat img, float sigma);
cv::Mat simplify_image(cv::Mat img, float limit, cv::Size grid);
std::vector<cv::Vec2f> find_lines(const cv::Mat& edges);
float normalize_angle(float angle);
bool is_horizontal(cv::Vec2f& line, float threshold);
bool is_vertical(cv::Vec2f& line, float threshold);
void split_lines_into_hv(std::vector<LineWrapper> &lines, std::vector<LineWrapper> &h_lines, std::vector<LineWrapper> &v_lines);
//void overlay_lines(cv::Mat& img, std::vector<cv::Vec2f>& lines, cv::Scalar color);
void overlay_lines(cv::Mat& img, std::vector<LineWrapper>& lines, const cv::Scalar& color);
void overlay_markers(cv::Mat& img, IntersectionsVec & points, const cv::Scalar& color);
void process_img(cv::Mat img, IntersectionsVec &intersections, cv::Mat &intersection_mat,
                 cv::Mat &simplified_image, cv::Mat &mozaic);
template<typename T> void trim_vector(std::vector<T>& v, int size);
std::vector<LineWrapper> remove_duplicate_lines(std::vector<LineWrapper> &lines);
bool are_duplicates(cv::Vec2f &line_a, cv::Vec2f &line_b, float rho_threshold, float theta_threshold);
std::vector<LineWrapper> wrap_lines(std::vector<cv::Vec2f> &lines);
std::vector<LineWrapper> remove_intersecting_lines(std::vector<LineWrapper> &lines, bool are_vertical);
bool intersect(const cv::Vec2f& line_a, const cv::Vec2f& line_b, cv::Vec2f &out);
bool are_intersecting_in_range(const cv::Vec2f& line_a, const cv::Vec2f& line_b, float xy_min, float xy_max);
std::vector<LineWrapper> remove_suspiciously_narrow_lines(std::vector<LineWrapper> line_wrappers, bool are_vertical, float accepted_min_width = 0.8, float accepted_max_width = 1.5);
IntersectionsVec segment_intersections(const std::vector<LineWrapper>& h_lines, const std::vector<LineWrapper>& v_lines);
cv::Vec2f create_vertical_line(float x1, float x2, float y2);
cv::Vec2f create_horizontal_line(float y1, float x2, float y2);
std::vector<LineWrapper> insert_missing_lines(std::vector<LineWrapper> &lines, float min_center_gap, bool are_vertical);
std::vector<LineWrapper> recalculate_wrappers_properties(std::vector<LineWrapper> &line_wrappers, bool are_vertical);
cv::Mat perspective_transform(cv::Mat &image, cv::Vec2f top_l, cv::Vec2f top_r, cv::Vec2f bottom_l, cv::Vec2f bottom_r);
bool split_image_into_folder(cv::Mat &image, IntersectionsVec &intersections, std::string path);
std::vector<cv::Mat> split_image_into_vector(cv::Mat &image, IntersectionsVec &intersections);

