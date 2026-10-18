# Camera tools for QNX on Raspberry Pi
#
# Cross-compile (after sourcing qnxsdp-env.sh):   make
# Build on the Pi (camera headers in ./camera):   make CXX=clang++ EXTRA_CXXFLAGS=-I.

CXX = q++ -Vgcc_ntoaarch64le
CXXFLAGS = -std=gnu++17 -O2 -Wall -Wextra $(EXTRA_CXXFLAGS)

all: frame_grab frame_stream

frame_grab: frame_grab.cpp camera_frame.cpp jpeg_encoder.cpp camera_frame.h jpeg_encoder.h
	$(CXX) $(CXXFLAGS) frame_grab.cpp camera_frame.cpp jpeg_encoder.cpp -o $@ -lcamapi

frame_stream: frame_stream.cpp camera_frame.cpp jpeg_encoder.cpp camera_frame.h jpeg_encoder.h
	$(CXX) $(CXXFLAGS) frame_stream.cpp camera_frame.cpp jpeg_encoder.cpp -o $@ -lcamapi -lsocket

clean:
	rm -f frame_grab frame_stream

.PHONY: all clean
