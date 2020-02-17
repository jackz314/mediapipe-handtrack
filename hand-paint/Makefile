all: build

build:
	bazel-1.2.1 build -c opt --copt -DMESA_EGL_NO_X11_HEADERS hand-paint
	mv -f ../bazel-bin/hand-paint/hand-paint ./hand-paint

run:
	cd .. && GLOG_logtostderr=1 hand-paint/hand-paint

debug:
	cd .. && gdb ./hand-paint/hand-paint

clean:
	rm -f ./hand-paint

clean-run: clean build run

clean-debug: clean build debug
