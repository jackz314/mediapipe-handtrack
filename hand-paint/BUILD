package(default_visibility = [
    "//visibility:public",
])

# Linux only.
# Must have a GPU with EGL support:
# ex: sudo aptitude install mesa-common-dev libegl1-mesa-dev libgles2-mesa-dev
# (or similar nvidia/amd equivalent)

cc_binary(
    name = "hand-paint",
    deps = [
        "//hand-paint:hand-paint-lib",
        "//mediapipe/graphs/hand_tracking:multi_hand_mobile_calculators",
    ],
)

cc_library(
    name = "hand-paint-lib",
    srcs = ["hand-paint.cc"],
    hdrs = ["rwqueue.h", "atomicops.h"],
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:commandlineflags",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_video",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "//mediapipe/gpu:gl_calculator_helper",
        "//mediapipe/gpu:gpu_buffer",
        "//mediapipe/gpu:gpu_shared_data_internal",
        "//mediapipe/framework/formats:landmark_cc_proto" ,
    ],
)
