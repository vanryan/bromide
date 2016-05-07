#include "io.h"

#include <fcntl.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/text_format.h>
#include <stdint.h>

#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <vector>

const int kProtoReadBytesLimit = INT_MAX;  // Max size of 2 GB minus 1 byte.

namespace Bromide {

  using google::protobuf::io::FileInputStream;
  using google::protobuf::io::FileOutputStream;
  using google::protobuf::io::ZeroCopyInputStream;
  using google::protobuf::io::CodedInputStream;
  using google::protobuf::io::ZeroCopyOutputStream;
  using google::protobuf::io::CodedOutputStream;
  using google::protobuf::Message;

  using std::ios;
  using std::fstream;
  using std::string;

  using caffe::SolverParameter;
  using caffe::NetParameter;

  bool ReadProtoFromTextFile(const string& filename, Message* proto) {
    int fd = open(filename.c_str(), O_RDONLY);
    CHECK_NE(fd, -1) << "File not found: " << filename;
    FileInputStream* input = new FileInputStream(fd);
    bool success = google::protobuf::TextFormat::Parse(input, proto);
    delete input;
    close(fd);
    return success;
  }

  void WriteProtoToTextFile(const Message& proto, const char* filename) {
    int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    FileOutputStream* output = new FileOutputStream(fd);
    google::protobuf::TextFormat::Print(proto, output);
    delete output;
    close(fd);
  }

  bool ReadProtoFromBinaryFile(const string& filename, Message* proto) {
    int fd = open(filename.c_str(), O_RDONLY);
    CHECK_NE(fd, -1) << "File not found: " << filename;
    ZeroCopyInputStream* raw_input = new FileInputStream(fd);
    CodedInputStream* coded_input = new CodedInputStream(raw_input);
    coded_input->SetTotalBytesLimit(kProtoReadBytesLimit, 536870912);

    bool success = proto->ParseFromCodedStream(coded_input);

    delete coded_input;
    delete raw_input;
    close(fd);
    return success;
  }

  void WriteProtoToBinaryFile(const Message& proto, const string& filename) {
    fstream output(filename, ios::out | ios::trunc | ios::binary);
    proto.SerializeToOstream(&output);
  }

  void ReadNetParamsFromTextFile(const string& param_file, NetParameter* param) {
    CHECK(ReadProtoFromTextFile(param_file, param))
        << "Failed to parse NetParameter file: " << param_file;
  }

  void ReadNetParamsFromBinaryFile(const string& param_file, NetParameter *param) {
    CHECK(ReadProtoFromBinaryFile(param_file, param))
        << "Failed to parse NetParameter file: " << param_file;
  }

  void ReadSolverParamsFromTextFile(const string& param_file, SolverParameter* param) {
    CHECK(ReadProtoFromTextFile(param_file, param))
        << "Failed to parse SolverParams file: " << param_file;
  }

}  // namespace Bromide
