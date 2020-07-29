CXX=clang++

DNN_DIR = $(PWD)
DNN_INCL = -I$(DNN_DIR)/include
CXXFLAGS = -std=c++17
CXXFLAGS += -O3 -g 
CXXFLAGS += `pkg-config --cflags --libs eigen3` $(DNN_INCL)

CXXSHARED = $(CXXFLAGS) -shared -fPIC
INSTALLDIR = $(DNN_DIR)
LIBTARGET = $(INSTALLDIR)/lib/libdnn.so
DNN_SRCS = src/Layer.cc src/Network.cc
default: all

all: $(LIBTARGET)

$(LIBTARGET): $(DNN_SRCS)
	$(CXX) $(CXXSHARED) $^ -o $@



ltest: tests/layertest.cpp
	$(CXX) $< $(CXXFLAGS) -o $@

ntest: tests/networktest.cpp
	$(CXX) $< $(CXXFLAGS) -o $@





