#include <opencv2/imgproc.hpp>
#include <iostream>
#include <utility>

#include "chessboard_detection.h"
//
// Created by eg4l on 10.02.2021.
//

float DEBUG_MEDIAN = 0.0f;

template<typename T>
void trim_vector(std::vector<T>& v, int size)
{
    if (v.size() > size)
    {
        v.resize(size);
    }
}

int median(cv::Mat& input)
{
    int rows = input.rows;
    int cols = input.cols;
    int histogram[256] = { 0 };
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
    cv::erode(edges, edges, cv::Mat(), cv::Point(-1, -1), 1);

    return edges;
}

cv::Mat simplify_image(cv::Mat img, float limit, cv::Size grid)
{
    cv::cvtColor(img, img, cv::COLOR_RGB2GRAY);
    cv::createCLAHE(limit,
                    std::move(grid))->apply(img, img);

    if (limit != 0.0f){
        cv::morphologyEx(img, img, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(10, 10)));
    }

    return img;
}

std::vector<cv::Vec2f> find_lines(const cv::Mat& edges)
{
    std::vector<cv::Vec2f> lines;
    cv::HoughLines(edges, lines, 1, M_PI/180.0f, 30);
    trim_vector(lines, 150);
    return lines;
}

float normalize_angle(float angle)
{
    return std::fmod(angle, M_PI);
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

void split_lines_into_hv(std::vector<LineWrapper> &lines, std::vector<LineWrapper> &h_lines, std::vector<LineWrapper> &v_lines)
{
    h_lines.reserve(50);
    v_lines.reserve(50);

    for(auto& line : lines)
    {
        if(is_horizontal(line.value, 12.0f*M_PI/180.0f))
        {
            if(h_lines.size() < 50)
                h_lines.push_back(line);
        }
        else if(is_vertical(line.value, 12.0f*M_PI/180.0f))
        {
            if(v_lines.size() < 50)
                v_lines.push_back(line);
        }
    }
}

void overlay_lines(cv::Mat& img, std::vector<LineWrapper>& lines, const cv::Scalar& color)
{
    for(const auto& line : lines)
    {
        auto[rho, theta] = line.value.val;
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

void overlay_markers(cv::Mat& img, std::vector<std::vector<cv::Vec2f>>& points, const cv::Scalar& color)
{
    for (auto& v : points)
        for(auto& point : v)
        {
            cv::drawMarker(img, cv::Point(point[0], point[1]), color, cv::MARKER_CROSS, 20, 2);
        }
}

bool are_duplicates(cv::Vec2f &line_a, cv::Vec2f &line_b, float rho_threshold, float theta_threshold)
{
    auto [rho_a, theta_a] = line_a.val;
    auto [rho_b, theta_b] = line_b.val;

    if (rho_a < 0)
    {
        rho_a = -rho_a;
        theta_a += M_PI;
    }

    if (rho_b < 0)
    {
        rho_b = -rho_b;
        theta_b += M_PI;
    }

    auto rho_diff = std::fmod(theta_a - theta_b + M_PI, 2*M_PI) - M_PI;

    return (std::abs(rho_diff) < theta_threshold && abs(rho_a - rho_b) < rho_threshold);
}

std::vector<LineWrapper> remove_duplicate_lines(std::vector<LineWrapper> &lines)
{
    std::vector<LineWrapper> result;
    result.reserve(lines.size());

    auto lines_size = lines.size();

    for (int i = 0; i < lines_size; i++)
    {
        auto& line = lines[i];
        if ( std::all_of(lines.begin(), lines.begin() + i, [&](auto other_line){return !are_duplicates(line.value, other_line.value, 5, 5.0f*M_PI/180.0f);}) )
        {
            result.push_back(line);
        }
    }

    return result;
}

bool intersect(const cv::Vec2f& line_a, const cv::Vec2f& line_b, cv::Vec2f& out)
{
    auto [rho1, theta1] = line_a.val;
    auto [rho2, theta2] = line_b.val;

    if (std::abs(theta1 - theta2) < 0.00001)
        return false;

    out[1] = (rho1*cos(theta2) - rho2*cos(theta1)) / (sin(theta1)*cos(theta2) - sin(theta2)*cos(theta1));
    out[0] = (rho1*sin(theta2) - rho2*sin(theta1)) / (cos(theta1)*sin(theta2) - cos(theta2)*sin(theta1));
    return true;
}

bool are_intersecting_in_range(const cv::Vec2f& line_a, const cv::Vec2f& line_b, float xy_min, float xy_max)
{
    cv::Vec2f intersection;
    if (!intersect(line_a, line_b, intersection))
        return false;

    return xy_min < intersection[0] and intersection[0] < xy_max and xy_min < intersection[1] and intersection[1] < xy_max;
}

std::vector<LineWrapper> remove_intersecting_lines(std::vector<LineWrapper> &lines, bool are_vertical)
{
    if (lines.empty())
        return std::vector<LineWrapper>();

    std::vector<LineWrapper> line_wrappers;
    line_wrappers.reserve(lines.size());
    for (auto& line : lines)
    {
        line_wrappers.push_back({line});
    }

    float min_pos = 0, max_pos = 512;

    line_wrappers = recalculate_wrappers_properties(line_wrappers, are_vertical);

    std::vector<LineWrapper> certain_lines_wrappers;
    certain_lines_wrappers.reserve(lines.size());

    auto line_wrapper = line_wrappers.begin();
    while(line_wrapper != line_wrappers.end())
    {
        bool is_certain = true;
        for(auto& other_line_wrapper : line_wrappers)
        {
            if(are_intersecting_in_range(line_wrapper->value, other_line_wrapper.value, min_pos - 128, max_pos+128))
            {
                is_certain = false;
                break;
            }
        }

        if(is_certain)
        {
            certain_lines_wrappers.push_back(*line_wrapper);
            line_wrapper = line_wrappers.erase(line_wrapper);
        }
        else
        {
            ++line_wrapper;
        }
    }

    std::sort(certain_lines_wrappers.begin(),
              certain_lines_wrappers.end(),
              [](const LineWrapper& line_a, const LineWrapper& line_b){return line_a.position_at_min > line_b.position_at_max;}
              );

    std::vector<LineWrapper> intersecting_lines;
    while (!line_wrappers.empty())
    {
        auto current_line = line_wrappers.begin();
        LineWrapper* prev_certain_line = nullptr;
        LineWrapper* next_certain_line = nullptr;

        for(auto &certain_line : certain_lines_wrappers)
        {
            if (certain_line.position_at_min > current_line->position_at_min)
            {
                next_certain_line = &certain_line;
                break;
            }
        }
        for(auto &certain_line : certain_lines_wrappers)
        {
            if (certain_line.position_at_min < current_line->position_at_min)
            {
                prev_certain_line = &certain_line;
                break;
            }
        }

        intersecting_lines.clear();
        auto other_line_wrapper = line_wrappers.begin();
        ++other_line_wrapper;
        while(other_line_wrapper != line_wrappers.end())
        {
            if(are_intersecting_in_range(current_line->value, other_line_wrapper->value, min_pos-128, max_pos+128))
            {
                intersecting_lines.push_back(*other_line_wrapper);
                other_line_wrapper = line_wrappers.erase(other_line_wrapper);
            }
            else
            {
                ++other_line_wrapper;
            }
        }
        intersecting_lines.push_back(*current_line);
        line_wrappers.erase(current_line);

        for (auto& line_wrapper : intersecting_lines)
        {
            if(next_certain_line && prev_certain_line)
            {
                line_wrapper.ratio_at_max = (next_certain_line->position_at_max - line_wrapper.position_at_max) / (next_certain_line->position_at_max - prev_certain_line->position_at_max);
                line_wrapper.ratio_at_min = (next_certain_line->position_at_min - line_wrapper.position_at_min) / (next_certain_line->position_at_min - prev_certain_line->position_at_min);
            }
            else if(auto neighbour_line = next_certain_line ? next_certain_line : prev_certain_line)
            {
                line_wrapper.ratio_at_min = neighbour_line->position_at_min - line_wrapper.position_at_min;
                line_wrapper.ratio_at_max = neighbour_line->position_at_max - line_wrapper.position_at_max;
            }
            else
            {
                throw std::runtime_error("Jan Pawłowski jest leniem\n");
            }
        }

        std::sort(intersecting_lines.begin(),
                  intersecting_lines.end(),
                  [](const LineWrapper& line_a, const LineWrapper& line_b){return abs(line_a.ratio_at_max - line_a.ratio_at_min) < abs(line_b.ratio_at_max - line_b.ratio_at_min);}
        );

        auto best_line = intersecting_lines[0];

        bool is_certain = true;
        for(auto& other_line_wrapper : line_wrappers)
        {
            if(are_intersecting_in_range(best_line.value, other_line_wrapper.value, min_pos - 128.0f, max_pos+128.0f))
            {
                is_certain = false;
                break;
            }
        }
        if(is_certain)
            certain_lines_wrappers.push_back(best_line);
        else
            line_wrappers.push_back(best_line);
    }

    return certain_lines_wrappers;
}

std::vector<LineWrapper> wrap_lines(std::vector<cv::Vec2f> &lines)
{
    std::vector<LineWrapper> result;
    result.reserve(lines.size());
    for(auto& line : lines)
        result.push_back({line});
    return result;
}

float max_non_inf(float a, float b)
{
    if(a == INFINITY)
        return b;
    if(b == INFINITY)
        return a;
    return std::max(a, b);
}

std::vector<LineWrapper> recalculate_wrappers_properties(std::vector<LineWrapper>& line_wrappers, bool are_vertical)
{
    for(auto& line_wrapper : line_wrappers)
    {
        if (are_vertical)
        {
            cv::Vec2f intersection;
            intersect(line_wrapper.value, cv::Vec2f(0, M_PI_2), intersection);
            line_wrapper.position_at_min = intersection[0];
            intersect(line_wrapper.value, cv::Vec2f(512, M_PI_2), intersection);
            line_wrapper.position_at_max = intersection[0];
        }
        else
        {
            cv::Vec2f intersection;
            intersect(line_wrapper.value, cv::Vec2f(0, 0.0f), intersection);
            line_wrapper.position_at_min = intersection[1];
            intersect(line_wrapper.value, cv::Vec2f(512, 0.0f), intersection);
            line_wrapper.position_at_max = intersection[1];
        }
    }

    for (auto &line_wrapper : line_wrappers)
    {
        line_wrapper.position_at_center = (line_wrapper.position_at_min + line_wrapper.position_at_max) / 2;
    }

    std::sort(line_wrappers.begin(),
              line_wrappers.end(),
              [](const LineWrapper& line_a, const LineWrapper& line_b){return line_a.position_at_center < line_b.position_at_center;}
    );

    for(int i = 1; i < line_wrappers.size(); i++)
    {
        line_wrappers[i-1].offset_to_next = line_wrappers[i].offset_from_prev = line_wrappers[i].position_at_center - line_wrappers[i - 1].position_at_center;
    }

    return line_wrappers;
}

std::vector<LineWrapper> remove_suspiciously_narrow_lines(std::vector<LineWrapper> line_wrappers, bool are_vertical, float accepted_min_width, float accepted_max_width)
{
    if (line_wrappers.size() < 2)
        return line_wrappers;

    line_wrappers = recalculate_wrappers_properties(line_wrappers, are_vertical);

    line_wrappers.erase(std::remove_if(line_wrappers.begin(), line_wrappers.end(), [](LineWrapper& line){return max_non_inf(line.offset_from_prev, line.offset_to_next) < 20;}), line_wrappers.end());

    line_wrappers = recalculate_wrappers_properties(line_wrappers, are_vertical);

    std::vector<float> gaps;
    gaps.reserve(line_wrappers.size()-1);

    for(int i = 1; i < line_wrappers.size(); i++)
        gaps.push_back(line_wrappers[i].offset_from_prev);

    std::sort(gaps.begin(),
              gaps.end(),
              std::greater<>()
    );

    float median_gap = gaps[gaps.size()/2];
    DEBUG_MEDIAN = median_gap;
//    std::cout << "Median gap is " << median_gap << "\n";

    std::vector<LineWrapper> result_line_wrappers;
    result_line_wrappers.reserve(line_wrappers.size());

    for(auto& line_wrapper : line_wrappers)
    {
        if((line_wrapper.offset_from_prev != INFINITY and accepted_min_width < line_wrapper.offset_from_prev/median_gap and line_wrapper.offset_from_prev/median_gap < accepted_max_width)
        or (line_wrapper.offset_to_next != INFINITY and accepted_min_width < line_wrapper.offset_to_next/median_gap and line_wrapper.offset_to_next/median_gap < accepted_max_width))
        {
            result_line_wrappers.push_back(line_wrapper);
        }
    }

    if (result_line_wrappers.size() > 9)
    {
        std::sort(result_line_wrappers.begin(),
                  result_line_wrappers.end(),
                  [](const LineWrapper& line_a, const LineWrapper& line_b){
                         return max_non_inf(line_a.offset_from_prev, line_a.offset_to_next) < max_non_inf(line_b.offset_from_prev, line_b.offset_to_next);
                    }
        );

        trim_vector(result_line_wrappers, 9);

        std::sort(result_line_wrappers.begin(),
                  result_line_wrappers.end(),
                  [](const LineWrapper& line_a, const LineWrapper& line_b){
                      return line_a.position_at_center > line_b.position_at_center;
                  }
        );
    }

    return result_line_wrappers;
}

std::vector<std::vector<cv::Vec2f>> segment_intersections(const std::vector<LineWrapper>& h_lines, const std::vector<LineWrapper>& v_lines)
{
    std::vector<std::vector<cv::Vec2f>> result;
    result.reserve(h_lines.size());

    for(auto& h_line : h_lines)
    {
        std::vector<cv::Vec2f> intersections;
        intersections.reserve(v_lines.size());

        for(auto& v_line : v_lines)
        {
            cv::Vec2f intersection;
            intersect(v_line.value, h_line.value, intersection);
            intersections.push_back(intersection);
        }
        result.push_back(intersections);
    }

    return result;
}

cv::Vec2f normalize_line(cv::Vec2f line)
{
    while (line[1] > M_PI)
    {
        line[0] *= -1;
        line[1] -= M_PI;
    }

    while (line[1] < 0)
    {
        line[0] *= -1;
        line[1] += M_PI;
    }

    return line;
}

cv::Vec2f create_vertical_line(float x1, float x2, float y2)
{
    float d = sqrt(y2*y2 + (x2 - x1)*(x2 - x1));
    float rho = y2 * x1 / d;
    float theta = acos(y2 / d);

    return normalize_line(cv::Vec2f(rho, theta));
}

cv::Vec2f create_horizontal_line(float y1, float x2, float y2)
{
    auto [rho, theta] = create_vertical_line(y1, y2, x2).val;
    return normalize_line(cv::Vec2f(M_PI_2 - rho, theta));
}

std::vector<LineWrapper> insert_missing_lines(std::vector<LineWrapper> &lines, float min_center_gap, bool are_vertical)
{
    if (lines.size() > 9)
    {
        throw std::runtime_error("More than 9 lines at this stage of processing is an error!");
    }
    if (lines.size() == 9)
    {
        return lines;
    }
    if (lines.size() < 3)
    {
        throw std::runtime_error("nothing i can do here");
    }

    while (lines.size() < 9)
    {
        lines = recalculate_wrappers_properties(lines, are_vertical);
        float median_gap = DEBUG_MEDIAN;
        auto max_offset_from_prev_line = std::max_element(lines.begin()+1,
                                                          lines.end(),
                                                          [](const LineWrapper& line_a, const LineWrapper& line_b){return line_a.offset_from_prev < line_b.offset_from_prev;}
                                                        );
        float max_gap = max_offset_from_prev_line->offset_from_prev;

        if (max_gap/median_gap > min_center_gap)
        {
            float max_point = ((max_offset_from_prev_line-1)->position_at_max + max_offset_from_prev_line->position_at_max) / 2;
            float min_point = ((max_offset_from_prev_line-1)->position_at_min + max_offset_from_prev_line->position_at_min) / 2;

            cv::Vec2f line;
            if(are_vertical)
                line = create_vertical_line(min_point, max_point, -512);
            else
                line = create_horizontal_line(min_point, max_point, -512);

            lines.push_back({line});
            lines = recalculate_wrappers_properties(lines, are_vertical);
        }
        else
        {
            lines = recalculate_wrappers_properties(lines, are_vertical);

            float min_center_position = lines.front().position_at_center;
            float max_center_position = lines.back().position_at_center;

            float min_point, max_point;
            if (min_center_position > 512 - max_center_position)
            {
                min_point = lines.front().position_at_min - median_gap;
                max_point = lines.front().position_at_max - median_gap;
            }
            else
            {
                min_point = lines.back().position_at_min + median_gap;
                max_point = lines.back().position_at_max + median_gap;
            }

            cv::Vec2f line;
            if(are_vertical)
                line = create_vertical_line(min_point, max_point, -512);
            else
                line = create_horizontal_line(min_point, max_point, -512);

            lines.push_back({line});
            lines = recalculate_wrappers_properties(lines, are_vertical);
        }
    }
    return lines;
}

#define RETURN_MOZAIC 0

cv::Mat process_img(cv::Mat img)
{
    cv::Mat temp;

    cv::resize(img, img, cv::Size(512, 512));

#if RETURN_MOZAIC == 1
    auto output_image0 = img.clone();
    auto output_image1 = img.clone();
#endif

    auto simplified_image = simplify_image(img, 3, cv::Size(2, 6));
    auto edges = auto_canny(simplified_image, 0.33f);

#if RETURN_MOZAIC == 1
    cv::cvtColor(edges, temp, cv::COLOR_GRAY2BGR);
    cv::hconcat(output_image0, temp, output_image0);
#endif

    auto lines = find_lines(edges);
    auto line_wrappers = wrap_lines(lines);

#if RETURN_MOZAIC == 1
    temp = img.clone();
    overlay_lines(temp, line_wrappers, cv::Scalar(255, 255, 255));
    cv::hconcat(output_image0, temp, output_image0);
#endif

    std::vector<LineWrapper> h_lines;
    std::vector<LineWrapper> v_lines;
    split_lines_into_hv(line_wrappers, h_lines, v_lines);

    h_lines = remove_duplicate_lines(h_lines);
    v_lines = remove_duplicate_lines(v_lines);

#if RETURN_MOZAIC == 1
    overlay_lines(output_image1, v_lines, cv::Scalar(0, 255, 0));
    overlay_lines(output_image1, h_lines, cv::Scalar(0, 0, 255));
#endif

    h_lines = remove_intersecting_lines(h_lines, false);
    v_lines = remove_intersecting_lines(v_lines, true);

    h_lines = remove_suspiciously_narrow_lines(h_lines, false, 0.75, 2.5);
    v_lines = remove_suspiciously_narrow_lines(v_lines, true, 0.75, 2.5);

#if RETURN_MOZAIC == 1
    temp = img.clone();
    overlay_lines(temp, v_lines, cv::Scalar(0, 255, 0));
    overlay_lines(temp, h_lines, cv::Scalar(0, 0, 255));
    cv::hconcat(output_image1, temp, output_image1);
#endif

    h_lines = insert_missing_lines(h_lines, 1.7f, false);
//    std::cout << "Before: " << v_lines.size();
    v_lines = insert_missing_lines(v_lines, 1.7f, true);
//    std::cout << " After: " << v_lines.size() << "\n";

    auto intersections = segment_intersections(h_lines, v_lines);

    temp = img.clone();
    overlay_lines(temp, v_lines, cv::Scalar(0, 255, 0));
    overlay_lines(temp, h_lines, cv::Scalar(0, 0, 255));
    overlay_markers(temp, intersections, cv::Scalar(255, 0, 0));

#if RETURN_MOZAIC == 1
    cv::hconcat(output_image1, temp, output_image1);
    cv::Mat output_image;
//    cv::copyMakeBorder(output_image1, output_image1, 0, 0, 0, 512, cv::BORDER_CONSTANT);
    cv::vconcat(output_image0, output_image1, output_image);
    return output_image;
#else
    return temp;
#endif

}
