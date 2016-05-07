g++ -c caffe.pb.cc -g -I ../include -lglog -lprotobuf -lpthread
g++ -c main.cpp -g -I ../include -lglog -lprotobuf -lpthread
g++ -o google_readLenet main.o caffe.pb.o -g -I ../include -lglog -lprotobuf -lpthread
LD_LIBRARY_PATH=/usr/local/lib ./google_readLenet