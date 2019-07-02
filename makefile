MLKROOT = /opt/intel/compilers_and_libraries_2019.4.233/mac/mkl

CXX=clang++

CXXFLAGS = -std=c++17
CXXFLAGS += -O3
CXXFLAGS += -I /usr/local/Cellar/eigen/3.3.7/include/eigen3

ltest: layertest.cpp
	$(CXX) layertest.cpp $(CXXFLAGS) -o ltest

ntest: networktest.cpp
	$(CXX) networktest.cpp $(CXXFLAGS) -o ntest





