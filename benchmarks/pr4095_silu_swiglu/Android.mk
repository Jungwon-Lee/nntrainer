LOCAL_PATH := $(call my-dir)
PR4095_BENCHMARK_PATH := $(LOCAL_PATH)

ifndef NNTRAINER_PACKAGE
$(error NNTRAINER_PACKAGE is not defined)
endif

include $(NNTRAINER_PACKAGE)/Android.mk
LOCAL_PATH := $(PR4095_BENCHMARK_PATH)

include $(CLEAR_VARS)

LOCAL_MODULE := layer_benchmark
LOCAL_SRC_FILES := layer_benchmark.cpp swiglu.cpp
LOCAL_C_INCLUDES := $(LOCAL_PATH)
LOCAL_CPPFLAGS := -O3 -std=c++17 -fexceptions -frtti
LOCAL_LDLIBS := -llog -landroid
LOCAL_SHARED_LIBRARIES := nntrainer

include $(BUILD_EXECUTABLE)
