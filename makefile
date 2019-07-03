CXX=clang++

CXXFLAGS = -std=c++17
CXXFLAGS += -O3
CXXFLAGS += `pkg-config --cflags --libs eigen3` -I include/boost

LIBS = -lomp
LIBS += -lboost_python -lboost_numpy


ltest: tests/layertest.cpp
	$(CXX) layertest.cpp $(CXXFLAGS) -o ltest

ntest: tests/networktest.cpp
	$(CXX) tests/networktest.cpp $(CXXFLAGS) -o ntest





