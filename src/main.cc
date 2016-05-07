#include "caffe.pb.h"
#include <stdio.h>
#include <string.h>
#include <fstream>
#include <iostream>
#include <glog/logging.h> 
#include <vector>
#include <string>

#include "Halide.h"

#include "bromide.h"

#include "CycleTimer.h"
#include "test_alexnet.h"
#include "test_lenet.h"


using namespace std;
using namespace caffe;
using namespace Bromide;



int main(int argc, char* argv[]) 
{ 
	google::InitGoogleLogging(argv[0]);
	
	test_lenet();

	return 0;
}
