cmake_minimum_required(VERSION 3.12)
project(simd_raw_data_compress VERSION 0.0.1)

find_package(daq-cmake REQUIRED)
find_package(Boost COMPONENTS unit_test_framework REQUIRED)
find_package(detdataformats REQUIRED)
find_package(CLI11 REQUIRED)
find_package(fdreadoutlibs REQUIRED)

daq_setup_environment()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -mavx2")

##############################################################################
# Main library

daq_add_library(LINK_LIBRARIES detdataformats::detdataformats fdreadoutlibs::fdreadoutlibs)


##############################################################################
# Plugins

##############################################################################
# Integration tests

##############################################################################
# Unit Tests

##############################################################################
# Applications
daq_add_application(simd_compress_benchmark simd_compress_benchmark.cxx TEST LINK_LIBRARIES simd_raw_data_compress CLI11::CLI11)

daq_install()
