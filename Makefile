CC = g++


IDIR = ./util \
	-I./src \
	-I~/halide/include \
	-I~/halide/tools \
	`pkg-config --cflags-only-I protobuf` \
	`pkg-config --cflags-only-I libglog`


CPPFLAGS  = --std=c++11 -g -fopenmp -Wall -I$(IDIR)


LFLAGS = -L ~/halide/bin
LFLAGS += -L ./external_lib

#SHELL := /bin/bash

LIBS = -lHalide -ldl

EXTRA_SCRIPTS = `pkg-config --libs protobuf libpng libglog`

SRCS = src/main.cc src/io.cc src/caffe.pb.cc

#OBJS = $(SRCS:.cc=.o)
#OBJS = src/main.o src/io.o src/caffe.pb.o src/test_alexnet.o
#OBJSS = src/main.o io.o caffe.pb.o test_alexnet.o
OBJS = src/main.o src/io.o src/caffe.pb.o
OBJSS = src/main.o io.o caffe.pb.o
MAIN = run_test

all: $(MAIN)
	@echo Bromide compiled!

####
src/io.o: src/io.cc
	$(CC) $(CPPFLAGS) -c src/io.cc $(LFLAGS) $(LIBS) $(EXTRA_SCRIPTS)

src/caffe.pb.o: src/caffe.pb.cc
	$(CC) $(CPPFLAGS) -c src/caffe.pb.cc $(LFLAGS) $(LIBS) $(EXTRA_SCRIPTS)

src/test_alexnet.o: src/test_alexnet.cc
	$(CC) $(CPPFLAGS) -c src/test_alexnet.cc $(LFLAGS) $(LIBS) $(EXTRA_SCRIPTS)
####



$(MAIN): $(OBJS)
	$(CC) $(CPPFLAGS) -o $(MAIN) $(OBJSS) $(LFLAGS) $(LIBS) $(EXTRA_SCRIPTS)

.PHONY: clean

clean:
	rm -f *.o src/*.o
