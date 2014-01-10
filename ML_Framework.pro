#-------------------------------------------------
#
# Project created by QtCreator 2013-09-28T00:21:50
#
#-------------------------------------------------

TEMPLATE += app
CONFIG -= qt gui app_bundle
CONFIG += console

TARGET = ML_Framework

TEMPLATE = app


SOURCES += main.cpp \
    CONV_NT.cpp \
    pso.cpp

#Include Paths
INCLUDEPATH += "F:\OpenCV2.4.1\opencv-built\install\include"


LIBS += "F:\OpenCV2.4.1\opencv-built\lib\libopencv_calib3d241.dll.a" \
"F:\OpenCV2.4.1\opencv-built\lib\libopencv_contrib241.dll.a" \
"F:\OpenCV2.4.1\opencv-built\lib\libopencv_core241.dll.a" \
"F:\OpenCV2.4.1\opencv-built\lib\libopencv_features2d241.dll.a" \
"F:\OpenCV2.4.1\opencv-built\lib\libopencv_flann241.dll.a" \
"F:\OpenCV2.4.1\opencv-built\lib\libopencv_gpu241.dll.a" \
"F:\OpenCV2.4.1\opencv-built\lib\libopencv_highgui241.dll.a" \
"F:\OpenCV2.4.1\opencv-built\lib\libopencv_imgproc241.dll.a" \
"F:\OpenCV2.4.1\opencv-built\lib\libopencv_legacy241.dll.a" \
"F:\OpenCV2.4.1\opencv-built\lib\libopencv_ml241.dll.a" \
"F:\OpenCV2.4.1\opencv-built\lib\libopencv_nonfree241.dll.a" \
"F:\OpenCV2.4.1\opencv-built\lib\libopencv_objdetect241.dll.a" \
"F:\OpenCV2.4.1\opencv-built\lib\libopencv_photo241.dll.a" \
"F:\OpenCV2.4.1\opencv-built\lib\libopencv_ts241.a" \
"F:\OpenCV2.4.1\opencv-built\lib\libopencv_video241.dll.a" \
"F:\OpenCV2.4.1\opencv-built\lib\libopencv_stitching241.dll.a" \
"F:\OpenCV2.4.1\opencv-built\lib\libopencv_videostab241.dll.a"

#Linker flags
QMAKE_LFLAGS +=

#Warnings
QMAKE_CFLAGS += -DASM_DEBUG # -DASM_FORCE_REFC
QMAKE_CFLAGS_RELEASE += -O3 -msse4.1 -mssse3 -msse3 -msse2 -msse -ffast-math -ftree-vectorize -ggdb3 -DSHOW_KPTS
QMAKE_CFLAGS_DEBUG += -ggdb3
QMAKE_CXXFLAGS += -fdiagnostics-show-option
QMAKE_CXXFLAGS_DEBUG += -ggdb3
QMAKE_CXXFLAGS_RELEASE += -O3 -msse4.1 -mssse3 -msse3 -msse2 -msse -ffast-math -ftree-vectorize -ggdb3 -DSHOW_KPTS
QMAKE_CXXFLAGS_WARN_OFF += -Wno-sign-compare -Wno-unused-function -Wno-unused-parameter
QMAKE_CXXFLAGS_WARN_ON  += -Wno-sign-compare -Wno-unused-function -Wno-unused-parameter

HEADERS += \
    ml_kernel.h \
    nn_mlp.h \
    CONV_NT.h \
    pso.h
