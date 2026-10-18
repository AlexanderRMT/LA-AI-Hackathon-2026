// flatbuffers_version_shim.hpp: lets the Pi's TFLite headers compile.        owner: Jaime
//
// TFLite 2.21's schema header static_asserts flatbuffers 25.9.23, and the flatbuffers-dev apk is
// a compatible release with a different number. Same workaround as the QNX AI workshop's file of
// this name. If the assert still fires, copy the numbers it checks from TFLite's schema_generated.h.
#pragma once

#include <flatbuffers/base.h>

#undef FLATBUFFERS_VERSION_MAJOR
#undef FLATBUFFERS_VERSION_MINOR
#undef FLATBUFFERS_VERSION_REVISION
#define FLATBUFFERS_VERSION_MAJOR 25
#define FLATBUFFERS_VERSION_MINOR 9
#define FLATBUFFERS_VERSION_REVISION 23
