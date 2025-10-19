#include <librealsense2/rs.hpp>
#include <opencv2/opencv.hpp>
#ifndef HEADLESS
#include <pangolin/pangolin.h>
#endif
#include <iostream>
#include <string>

// Utility Functions

cv::Mat get_color_frame(const rs2::frameset &frames) {
    rs2::video_frame color_frame = frames.get_color_frame();
    if (!color_frame) return {};
    return cv::Mat(cv::Size(color_frame.get_width(), color_frame.get_height()),
                   CV_8UC3, (void*)color_frame.get_data(), cv::Mat::AUTO_STEP).clone();
}

cv::Mat get_depth_frame(const rs2::frameset &frames) {
    rs2::depth_frame depth_frame = frames.get_depth_frame();
    if (!depth_frame) return {};
    return cv::Mat(cv::Size(depth_frame.get_width(), depth_frame.get_height()),
                   CV_16UC1, (void*)depth_frame.get_data(), cv::Mat::AUTO_STEP).clone();
}

cv::Mat convert_depth_to_8bit(const cv::Mat &depth, float scale = 255.0/4000.0f) {
    cv::Mat depth_8bit;
    depth.convertTo(depth_8bit, CV_8UC1, scale);
    return depth_8bit;
}

#ifndef HEADLESS
void render_pangolin_frame(cv::Mat &color, cv::Mat &depth, pangolin::View *display,
                           pangolin::GlTexture &color_tex, pangolin::GlTexture &depth_tex,
                           pangolin::OpenGlRenderState &cam_state) 
{
    if (!display) return;

    color_tex.Upload(color.data, GL_BGR, GL_UNSIGNED_BYTE);
    depth_tex.Upload(convert_depth_to_8bit(depth).data, GL_LUMINANCE, GL_UNSIGNED_BYTE);

    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    display->Activate(cam_state);
    color_tex.RenderToViewportFlipY();
    pangolin::FinishFrame();
}
#endif


// Main Program

int main(int argc, char** argv) {
    try {
        const bool has_display = getenv("DISPLAY") != nullptr;

#ifndef HEADLESS
        pangolin::OpenGlRenderState cam_state;
        pangolin::Handler3D handler(cam_state);
        pangolin::View* display_ptr = nullptr;
        pangolin::GlTexture color_tex, depth_tex;

        if (has_display) {
            std::cout << "[Pangolin] Display detected. Initializing window...\n";
            pangolin::CreateWindowAndBind("Pangolin RGB-D Test", 640, 480);
            glEnable(GL_DEPTH_TEST);

            cam_state = pangolin::OpenGlRenderState(
                pangolin::ProjectionMatrix(640,480,420,420,320,240,0.1,1000),
                pangolin::ModelViewLookAt(0,0,-3, 0,0,0, pangolin::AxisY)
            );

            display_ptr = &pangolin::CreateDisplay()
                .SetBounds(0.0,1.0,0.0,1.0,-640.0f/480.0f)
                .SetHandler(&handler);

            color_tex.Reinitialise(640, 480, GL_RGB);
            depth_tex.Reinitialise(640, 480, GL_LUMINANCE);

            cv::namedWindow("OpenCV Color Frame", cv::WINDOW_AUTOSIZE);
        }
#endif


        // RealSense pipeline setup
 
        rs2::pipeline pipe;
        rs2::config cfg;
        cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
        cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
        pipe.start(cfg);
        std::cout << "[RealSense] Pipeline started successfully.\n";

        cv::Mat last_color, last_depth;
        const int frame_count = 10;

        for (int i = 0; i < frame_count; ++i) {
            rs2::frameset frames = pipe.wait_for_frames();
            cv::Mat color = get_color_frame(frames);
            cv::Mat depth = get_depth_frame(frames);

            if (color.empty() || depth.empty()) {
                std::cerr << "[Warning] Invalid frame received. Skipping...\n";
                continue;
            }

            last_color = color;
            last_depth = depth;

            // Print frame statistics
            cv::Scalar mean_color = cv::mean(color);
            double mean_depth = cv::mean(depth)[0];
            std::cout << "[Frame " << i << "] Color mean: "
                      << mean_color[0] << ", "
                      << mean_color[1] << ", "
                      << mean_color[2] << " | Depth mean: "
                      << mean_depth << std::endl;

#ifndef HEADLESS
            if (has_display) {
                cv::imshow("OpenCV Color Frame", last_color);
                cv::waitKey(1);
                render_pangolin_frame(last_color, last_depth, display_ptr, color_tex, depth_tex, cam_state);
            }
#endif
        }

        std::cout << "[RealSense] Frame capture finished.\n";

#ifndef HEADLESS
        if (has_display) {
            std::cout << "[Pangolin/OpenCV] Entering persistent display loop. Close window to exit.\n";
            while (!pangolin::ShouldQuit()) {
                if (!last_color.empty()) {
                    cv::imshow("OpenCV Color Frame", last_color);
                    render_pangolin_frame(last_color, last_depth, display_ptr, color_tex, depth_tex, cam_state);
                    cv::waitKey(1);
                }
            }
            cv::destroyAllWindows();
        }
#else
        if (!last_color.empty()) cv::imwrite("last_color_frame.png", last_color);
        if (!last_depth.empty()) cv::imwrite("last_depth_frame.png", last_depth);
        std::cout << "[Headless] Saved last_color_frame.png and last_depth_frame.png\n";
#endif

    } catch (const rs2::error & e) {
        std::cerr << "[RealSense ERROR] " << e.what() << std::endl;
        return EXIT_FAILURE;
    } catch (const std::exception & e) {
        std::cerr << "[EXCEPTION] " << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
