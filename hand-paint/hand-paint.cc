// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// An example of sending OpenCV webcam frames into a MediaPipe graph.
// This example requires a linux computer and a GPU with EGL support drivers.

// Everything in this file should reference directory mediapipe as root, otherwise dependencies on files might not work

#include <cstdlib>

#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/commandlineflags.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/gpu/gl_calculator_helper.h"
#include "mediapipe/gpu/gpu_buffer.h"
#include "mediapipe/gpu/gpu_shared_data_internal.h"

// landmark stuff
#include "mediapipe/framework/formats/landmark.pb.h"

constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "output_video";
constexpr char kLandmarksStream[] = "multi_hand_landmarks";
constexpr char kWindowName[] = "Model Output";
constexpr char kStatWindowName[] = "Stats";
constexpr char kCanvasWindowName[] = "Canvas";

constexpr int canvas_height = 750;
constexpr int canvas_width = 1000;

DEFINE_string(input_video_path, "",
              "Full path of video to load. "
              "If not provided, attempt to use a webcam.");
DEFINE_string(output_video_path, "",
              "Full path of where to save result (.mp4 only). "
              "If not provided, show result in a window.");

DEFINE_double(zoom_level, 1.5, "Zoom level of output video, default 1.5");

std::string hand_tracking_graph_file = "hand-paint/multi_hand_tracking_mobile.pbtxt";

//helper function to display multi-line strings on opencv Mat
void displayMultilineString(std::stringstream& str, cv::Mat &img, 
    cv::HersheyFonts font_style = cv::FONT_HERSHEY_PLAIN, cv::Scalar text_color = cv::Scalar(0,0,0), int text_thickness = 1) {
  int y0 = 50, dy = 20;
  std::string line;
  int i = 0;
  while(std::getline(str, line)){
    int y = y0 + i*dy;
    cv::putText(img, line, cv::Point(50, y ), font_style, 1.2, text_color, text_thickness);
    ++i;
  }
}

void paintPts(const std::deque<cv::Point>& ptsQ, cv::Mat& canvas) {
  std::deque<cv::Point>::const_iterator it = ptsQ.begin();
  while(it != ptsQ.end()){
    cv::circle(canvas, *it, 5, cv::Scalar(0,0,0), 2);
    it++;
  }
}

::mediapipe::Status RunMPPGraph() {
  std::string calculator_graph_config_contents;
  MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
      hand_tracking_graph_file, &calculator_graph_config_contents));
  LOG(INFO) << "Get calculator graph config contents: "
            << calculator_graph_config_contents;
  mediapipe::CalculatorGraphConfig config =
      mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
          calculator_graph_config_contents);

  LOG(INFO) << "Initialize the calculator graph.";
  mediapipe::CalculatorGraph graph;
  MP_RETURN_IF_ERROR(graph.Initialize(config));

  LOG(INFO) << "Initialize the GPU.";
  ASSIGN_OR_RETURN(auto gpu_resources, mediapipe::GpuResources::Create());
  MP_RETURN_IF_ERROR(graph.SetGpuResources(std::move(gpu_resources)));
  mediapipe::GlCalculatorHelper gpu_helper;
  gpu_helper.InitializeForTest(graph.GetGpuResources().get());

  LOG(INFO) << "Initialize the camera or load the video.";
  cv::VideoCapture capture;
  const bool load_video = !FLAGS_input_video_path.empty();
  if (load_video) {
    capture.open(FLAGS_input_video_path);
  } else {
    capture.open(0);
  }
  RET_CHECK(capture.isOpened());

  cv::VideoWriter writer;
  const bool save_video = !FLAGS_output_video_path.empty();
  if (!save_video) {
    cv::namedWindow(kWindowName, /*flags=WINDOW_AUTOSIZE*/ 1);
    cv::moveWindow(kWindowName, 0, 0);

  // Camera settings, don't change
#if (CV_MAJOR_VERSION >= 3) && (CV_MINOR_VERSION >= 2)
    capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    capture.set(cv::CAP_PROP_FPS, 30);
#endif
  }

  //stats window
  cv::namedWindow(kStatWindowName, 1);
  cv::moveWindow(kStatWindowName, 1100, 0);//on the right side
  cv::Mat stat_bg(1000, 750, CV_8UC3, cv::Scalar(255,255,255)); // white background
  cv::imshow(kStatWindowName, stat_bg);

  //canvas window
  cv::namedWindow(kCanvasWindowName, 1);
  cv::Mat canvas_bg(canvas_height, canvas_width, CV_8UC3, cv::Scalar(255,255,255)); // white background  
  cv::imshow(kCanvasWindowName, canvas_bg);

  std::deque<cv::Point> ptsQ;

  LOG(INFO) << "Start running the calculator graph.";
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller,
                   graph.AddOutputStreamPoller(kOutputStream));

  // hand landmarks stream
  ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller landmark_poller,
            graph.AddOutputStreamPoller(kLandmarksStream));

  MP_RETURN_IF_ERROR(graph.StartRun({}));

  LOG(INFO) << "Start grabbing and processing frames.";
  bool grab_frames = true;
  while (grab_frames) {
    // Capture opencv camera or video frame.
    cv::Mat camera_frame_raw;
    capture >> camera_frame_raw;
    if (camera_frame_raw.empty()) break;  // End of video.
    cv::Mat camera_frame;
    cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);
    if (!load_video) {
      cv::flip(camera_frame, camera_frame, /*flipcode=HORIZONTAL*/ 1);
    }

    // Wrap Mat into an ImageFrame.
    auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
        mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
        mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
    cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
    camera_frame.copyTo(input_frame_mat);

    // Prepare and add graph input packet.
    size_t frame_timestamp_us =
        (double)cv::getTickCount() / (double)cv::getTickFrequency() * 1e6;
    MP_RETURN_IF_ERROR(
        gpu_helper.RunInGlContext([&input_frame, &frame_timestamp_us, &graph,
                                   &gpu_helper]() -> ::mediapipe::Status {
          // Convert ImageFrame to GpuBuffer.
          auto texture = gpu_helper.CreateSourceTexture(*input_frame.get());
          auto gpu_frame = texture.GetFrame<mediapipe::GpuBuffer>();
          glFlush();
          texture.Release();
          // Send GPU image packet into the graph.
          MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
              kInputStream, mediapipe::Adopt(gpu_frame.release())
                                .At(mediapipe::Timestamp(frame_timestamp_us))));
          return ::mediapipe::OkStatus();
        }));

    // Get the graph result packet, or stop if that fails.
    mediapipe::Packet packet;
    if (!poller.Next(&packet)) break;
    std::unique_ptr<mediapipe::ImageFrame> output_frame;

    mediapipe::Packet landmark_packet;
    if (!landmark_poller.Next(&landmark_packet)) break;

    //all hands
    int hand_index = 0;
    const auto& landmarks = landmark_packet.Get<std::vector<mediapipe::NormalizedLandmarkList>>();
    
    //stat stuff
    cv::Mat tmp_bg = stat_bg.clone();
    std::stringstream stat_str;
    stat_str << "++++++++\n";

    //canvas stuff
    cv::Mat canvas = canvas_bg.clone();

    for (const auto& landmark_list : landmarks) {
      // std::cout << landmark_list.DebugString();

      //lists of landmarks of individual hands
      int landmark_index = 0;
      for (const auto& landmark : landmark_list.landmark()) {
        std::ostringstream line_stream;
        line_stream << "[Hand<" << hand_index << ">] Landmark<" << landmark_index++ << ">: (" << landmark.x() << ", " << landmark.y() << ", " << landmark.z() << ")";
        stat_str << line_stream.str() << "\n";
        std::cout << line_stream.str() << "                         " << "\r";
        
        //canvas
        // int x_loc = landmark.x()*canvas_width;
        // int y_loc = landmark.y()*canvas_height;
        // cv::putText(canvas, std::to_string(landmark_index - 1), cv::Point(x_loc, y_loc), cv::FONT_HERSHEY_PLAIN, 1, cv::Scalar(0,0,0));
      }
      //canvas for individual hand
      const auto& landmark = landmark_list.landmark()[8];
      int x_loc = landmark.x()*canvas_width;
      int y_loc = landmark.y()*canvas_height;
      ptsQ.push_back(cv::Point(x_loc, y_loc));
      if(ptsQ.size() > 10){
        ptsQ.pop_front();
      }
      paintPts(ptsQ, canvas);
      // canvas.at<cv::Vec3b>(x_loc, y_loc) = cv::Vec3b(0,0,0);//change to black
      // cv::circle(canvas, cv::Point(x_loc, y_loc), 5, cv::Scalar(0,0,0), 2);
      // std::cout << "DRAW:" << x_loc << "," << y_loc << "\r";
      ++hand_index;
    }
    stat_str << "--------\n";
    // std::cout << stat_str.str() << std::endl;
    displayMultilineString(stat_str, tmp_bg);
    // cv::putText(tmp_bg, stat_str.str(), cv::Point(10,100), cv::FONT_HERSHEY_SIMPLEX, 1, cv::Scalar(0,0,0), 2);
    cv::imshow(kStatWindowName, tmp_bg);

    cv::imshow(kCanvasWindowName, canvas);

    // Convert GpuBuffer to ImageFrame.
    MP_RETURN_IF_ERROR(gpu_helper.RunInGlContext(
        [&packet, &output_frame, &gpu_helper]() -> ::mediapipe::Status {
          auto& gpu_frame = packet.Get<mediapipe::GpuBuffer>();
          auto texture = gpu_helper.CreateSourceTexture(gpu_frame);
          output_frame = absl::make_unique<mediapipe::ImageFrame>(
              mediapipe::ImageFormatForGpuBufferFormat(gpu_frame.format()),
              gpu_frame.width(), gpu_frame.height(),
              mediapipe::ImageFrame::kGlDefaultAlignmentBoundary);
          gpu_helper.BindFramebuffer(texture);
          const auto info =
              mediapipe::GlTextureInfoForGpuBufferFormat(gpu_frame.format(), 0);
          glReadPixels(0, 0, texture.width(), texture.height(), info.gl_format,
                       info.gl_type, output_frame->MutablePixelData());
          glFlush();
          texture.Release();
          return ::mediapipe::OkStatus();
        }));

    // Convert back to opencv for display or saving.
    cv::Mat output_frame_mat = mediapipe::formats::MatView(output_frame.get());
    cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);

    //zoom
    cv::resize(output_frame_mat, output_frame_mat, cv::Size(), FLAGS_zoom_level, FLAGS_zoom_level);

    if (save_video) {
      if (!writer.isOpened()) {
        LOG(INFO) << "Prepare video writer.";
        writer.open(FLAGS_output_video_path,
                    mediapipe::fourcc('a', 'v', 'c', '1'),  // .mp4
                    capture.get(cv::CAP_PROP_FPS), output_frame_mat.size());
        RET_CHECK(writer.isOpened());
      }
      writer.write(output_frame_mat);
    } else {
      cv::imshow(kWindowName, output_frame_mat);
      // Press any key to exit.
      // const int pressed_key = cv::waitKey(5);
      // if (pressed_key >= 0 && pressed_key != 255) grab_frames = false;

      // Press q to exit
      const int pressed_key = cv::waitKeyEx(5);
      if (pressed_key == 'q') grab_frames = false;
    }
  }

  LOG(INFO) << "Shutting down.";
  if (writer.isOpened()) writer.release();
  MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
  return graph.WaitUntilDone();
}

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  ::mediapipe::Status run_status = RunMPPGraph();
  if (!run_status.ok()) {
    LOG(ERROR) << "Failed to run the graph: " << run_status.message();
    return EXIT_FAILURE;
  } else {
    LOG(INFO) << "Success!";
  }
  return EXIT_SUCCESS;
}
