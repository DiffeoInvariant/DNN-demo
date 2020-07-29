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

DNN_LDFLAGS = -L$(INSTALLDIR)/lib -Wl,-rpath,$(INSTALLDIR)/lib -ldnn

.PHONY: all $(LIBTARGET) ltest ntest

default: all

all: $(LIBTARGET) ltest ntest

$(LIBTARGET): $(DNN_SRCS)
	$(CXX) $(CXXSHARED) $^ -o $@



ltest: tests/layertest.cpp
	$(CXX) $< $(CXXFLAGS) -o $@ $(DNN_LDFLAGS)

ntest: tests/networktest.cpp
	$(CXX) $< $(CXXFLAGS) -o $@ $(DNN_LDFLAGS)





