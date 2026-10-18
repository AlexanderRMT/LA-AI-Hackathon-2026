# Camera tools for QNX on Raspberry Pi
#
# Cross-compile (after sourcing qnxsdp-env.sh):   make
# Build on the Pi (camera headers in ./camera):   make CXX=clang++ EXTRA_CXXFLAGS=-I.
#
# The Mirage vision process needs OpenCV and TFLite, which only the Pi has, so build it there:
#   sudo apk add opencv-dev tflite-runtime tflite-runtime-dev flatbuffers-dev qnx-sf-base-dev
#   make mirage_vision CXX=clang++ EXTRA_CXXFLAGS=-I.

CXX = q++ -Vgcc_ntoaarch64le
CXXFLAGS = -std=gnu++17 -O2 -Wall -Wextra $(EXTRA_CXXFLAGS)

VISION = mirage-starter/vision
VISION_SRCS = $(VISION)/mirage_vision.cpp $(VISION)/mirage_hook.cpp $(VISION)/bos_engine.cpp $(VISION)/web_view.cpp camera_frame.cpp jpeg_encoder.cpp
OPENCV_CXXFLAGS = $(shell pkg-config --cflags opencv4 2>/dev/null || echo -I/usr/include/opencv4)
VISION_LIBS = -lopencv_video -lopencv_imgcodecs -lopencv_imgproc -lopencv_core -ltensorflow-lite

all: frame_grab frame_stream

frame_grab: frame_grab.cpp camera_frame.cpp jpeg_encoder.cpp camera_frame.h jpeg_encoder.h
	$(CXX) $(CXXFLAGS) frame_grab.cpp camera_frame.cpp jpeg_encoder.cpp -o $@ -lcamapi

frame_stream: frame_stream.cpp camera_frame.cpp jpeg_encoder.cpp camera_frame.h jpeg_encoder.h
	$(CXX) $(CXXFLAGS) frame_stream.cpp camera_frame.cpp jpeg_encoder.cpp -o $@ -lcamapi -lsocket

mirage_vision: $(VISION_SRCS) $(wildcard $(VISION)/*.hpp) mirage-starter/common/mirage_msgs.h camera_frame.h jpeg_encoder.h
	$(CXX) $(CXXFLAGS) $(OPENCV_CXXFLAGS) $(VISION_SRCS) -o $@ $(VISION_LIBS) -lcamapi -lsocket

clean:
	rm -f frame_grab frame_stream mirage_vision

.PHONY: all clean
