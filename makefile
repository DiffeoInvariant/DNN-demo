MLKROOT = /opt/intel/compilers_and_libraries_2019.4.233/mac/mkl

CXX=clang++

CXXFLAGS = -std=c++17
CXXFLAGS += -O3
CXXFLAGS += `pkg-config --cflags --libs eigen3` -lomp

ltest: tests/layertest.cpp
	$(CXX) layertest.cpp $(CXXFLAGS) -o ltest

ntest: tests/networktest.cpp
	$(CXX) tests/networktest.cpp $(CXXFLAGS) -o ntest





