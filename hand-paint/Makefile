all:
	bazel-1.2.1 build -c opt --copt -DMESA_EGL_NO_X11_HEADERS hand-paint

run:
	cd .. && GLOG_logtostderr=1 bazel-bin/hand-paint/hand-paint --calculator_graph_config_file=mediapipe/graphs/hand_tracking/multi_hand_tracking_mobile.pbtxt