#include <opencv2/imgproc.hpp>
#include <iostream>

#include "chessboard_detection.h"
//
// Created by eg4l on 10.02.2021.
//

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

    if (limit != 0.0f){
        cv::morphologyEx(img, img, cv::MORPH_CLOSE, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(10, 10)));
    }

    return img;
}

void find_lines(cv::Mat edges, std::vector<cv::Vec2f>& lines)
{
    cv::HoughLines(edges, lines, 1, M_PI/180.0f, 30);
    trim_vector(lines, 200);
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

void overlay_lines(cv::Mat& img, std::vector<LineWrapper>& lines, cv::Scalar color)
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

std::vector<cv::Vec2f> remove_duplicate_lines(std::vector<cv::Vec2f>& lines)
{
    std::vector<cv::Vec2f> result;
    result.reserve(lines.size());

    auto lines_size = lines.size();

    for (int i = 0; i < lines_size; i++)
    {
        auto& line = lines[i];
        if ( std::all_of(lines.begin(), lines.begin() + i, [&](cv::Vec2f other_line){return !are_duplicates(line, other_line, 5, 5*M_PI/180);}) )
        {
            result.push_back(line);
        }
    }

    return result;
}

bool intersect(cv::Vec2f line_a, cv::Vec2f line_b, cv::Vec2f& out)
{
    auto [rho1, theta1] = line_a.val;
    auto [rho2, theta2] = line_b.val;

    if (std::abs(theta1 - theta2) < 0.00001)
        return false;

    out[1] = (rho1*cos(theta2) - rho2*cos(theta1)) / (sin(theta1)*cos(theta2) - sin(theta2)*cos(theta1));
    out[0] = (rho1*sin(theta2) - rho2*sin(theta1)) / (cos(theta1)*sin(theta2) - cos(theta2)*sin(theta1));
    return true;
}

bool are_intersecting_in_range(cv::Vec2f line_a, cv::Vec2f line_b, float xy_min, float xy_max)
{
    cv::Vec2f intersection;
    if (!intersect(line_a, line_b, intersection))
        return false;

    return xy_min < intersection[0] and intersection[0] < xy_max and xy_min < intersection[1] and intersection[1] < xy_max;
}

std::vector<LineWrapper> remove_intersecting_lines(std::vector<cv::Vec2f>& lines, bool are_vertical)
{
//    if lines.shape == 0 or lines.shape[0] == 0:
//        return lines

    if (lines.empty())
    {
        return std::vector<LineWrapper>();
    }

//    line_wrappers = []
//    for line in lines:
//        line_wrappers.append(NpArrayWrapper(line))
//
    std::vector<LineWrapper> line_wrappers;
    line_wrappers.reserve(lines.size());
    for (auto& line : lines)
    {
        line_wrappers.push_back({line});
    }

//    min_pos = 0
//    max_pos = 512
//    for current_line in line_wrappers:
//        if are_vertical:
//            current_line.position_at_min = intersection(current_line.value, [[min_pos, np.pi/2]])[0]
//            current_line.position_at_max = intersection(current_line.value, [[max_pos, np.pi/2]])[0]
//        else:
//            current_line.position_at_min = intersection(current_line.value, [[min_pos, 0.0]])[1]
//            current_line.position_at_max = intersection(current_line.value, [[max_pos, 0.0]])[1]

    int min_pos = 0, max_pos = 512;

    for(auto& current_line_wrapper : line_wrappers)
    {
        if (are_vertical)
        {
            cv::Vec2f intersection;
            intersect(current_line_wrapper.value, cv::Vec2f(min_pos, M_PI_2), intersection);
            current_line_wrapper.position_at_min = intersection[0];
            intersect(current_line_wrapper.value, cv::Vec2f(max_pos, M_PI_2), intersection);
            current_line_wrapper.position_at_max = intersection[0];
        }
        else
        {
            cv::Vec2f intersection;
            intersect(current_line_wrapper.value, cv::Vec2f(min_pos, 0.0f), intersection);
            current_line_wrapper.position_at_min = intersection[1];
            intersect(current_line_wrapper.value, cv::Vec2f(max_pos, 0.0f), intersection);
            current_line_wrapper.position_at_max = intersection[1];
        }
    }


//    for index, current_line in enumerate(line_wrappers):
//        if all([
//            not are_intersecting_in_range(current_line.value, other.value, (min_pos-128, min_pos-128), (max_pos+128, max_pos+128)) for other in line_wrappers if other is not current_line
//        ]):
//            certain_lines.append(current_line)
//
//    for line_wrapper in certain_lines:
//        line_wrappers.remove(line_wrapper)
//

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

//    std::cout << "There are " << certain_lines_wrappers.size() << " certaing lines\n";

//    for(auto& certain_line_wrapper : certain_lines_wrappers)
//    {
//        auto index = line_wrappers.
//        line_wrappers.erase()
//    }


//    auto predicate = [&](LineWrapper &line_wrapper) {
//        return std::all_of(line_wrappers.begin(), line_wrappers.begin(), [&](LineWrapper other_line){return (&line_wrapper == &other_line) || !are_intersecting_in_range(line_wrapper.value, other_line.value, [min_pos-128, min_pos-128], [max_pos+128, max_pos+128]);});
//    };
//    line_wrappers.erase(std::remove_if(line_wrappers.begin(), line_wrappers.end(), predicate), line_wrappers.end());

//    certain_lines = sorted(certain_lines, key=lambda x: x.position_at_min)

    std::sort(certain_lines_wrappers.begin(),
              certain_lines_wrappers.end(),
              [](LineWrapper line_a, LineWrapper line_b){return line_a.position_at_min > line_b.position_at_max;}
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

//
//    while line_wrappers:
//        current_line = line_wrappers[0]
//        prev_certain_line = None
//        next_certain_line = None
//
//        for certain_line in certain_lines:
//            if certain_line.position_at_min > current_line.position_at_min:
//                next_certain_line = certain_line
//                break
//
//        for certain_line in reversed(certain_lines):
//            if certain_line.position_at_min < current_line.position_at_min:
//                prev_certain_line = certain_line
//                break
//
//        intersecting_lines = [
//            other for other in line_wrappers[1:] if are_intersecting_in_range(current_line.value, other.value, (min_pos-128, min_pos-128), (max_pos+128, max_pos+128))
//        ]
//        intersecting_lines.append(current_line)
//
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
                throw std::runtime_error("Jan Pawłowski jest leniem");
            }
        }


        std::sort(intersecting_lines.begin(),
                  intersecting_lines.end(),
                  [](LineWrapper line_a, LineWrapper line_b){return abs(line_a.ratio_at_max - line_a.ratio_at_min) < abs(line_b.ratio_at_max - line_b.ratio_at_min);}
        );
//        for line_wrapper in intersecting_lines:
//            if next_certain_line and prev_certain_line:
//                line_wrapper.ratio_at_max = (next_certain_line.position_at_max - line_wrapper.position_at_max) / (next_certain_line.position_at_max - prev_certain_line.position_at_max)
//                line_wrapper.ratio_at_min = (next_certain_line.position_at_min - line_wrapper.position_at_min) / (next_certain_line.position_at_min - prev_certain_line.position_at_min)
//            elif neighbour_line := next_certain_line or prev_certain_line:
//                line_wrapper.ratio_at_min = neighbour_line.position_at_min - line_wrapper.position_at_min
//                line_wrapper.ratio_at_max = neighbour_line.position_at_max - line_wrapper.position_at_max
//            else:
//                pass
//                # raise Exception("Programiście jeszcze nie chciało się pomyśleć co wtedy")
//
//        intersecting_lines = sorted(intersecting_lines, key=lambda x: abs(x.ratio_at_max - x.ratio_at_min))
//

        auto best_line = intersecting_lines[0];
        certain_lines_wrappers.push_back(best_line);
    }

    return certain_lines_wrappers;
//        best_line = intersecting_lines[0]
//        certain_lines.append(best_line)
//        for line_wrapper in intersecting_lines:
//            line_wrappers.remove(line_wrapper)
//
//    result = [line_wrapper.value for line_wrapper in certain_lines]
//    return result
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

    h_lines = remove_duplicate_lines(h_lines);
    v_lines = remove_duplicate_lines(v_lines);

    auto h_line_wrappers = remove_intersecting_lines(h_lines, false);
    auto v_line_wrappers = remove_intersecting_lines(v_lines, true);

    overlay_lines(img, v_line_wrappers, cv::Scalar(0, 255, 0));
    overlay_lines(img, h_line_wrappers, cv::Scalar(0, 0, 255));

    return img;
}
