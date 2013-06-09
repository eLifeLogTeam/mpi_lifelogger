cmake_minimum_required(VERSION 2.6)
project(MPI_Demo)
add_executable(MPI_Demo MPI_Demo/source/main.cpp)
add_library(OmniStitcher  OmniStitcher/source/OmniStitcherR)
