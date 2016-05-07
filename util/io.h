#ifndef BROMIDE_IO_H_
#define BROMIDE_IO_H_

#include "util.h"

namespace Bromide {

	using google::protobuf::Message;
	using std::string;
	using caffe::NetParameter;
	using caffe::SolverParameter;

	bool ReadProtoFromTextFile(const string& filename, Message* proto);
	void WriteProtoToTextFile(const Message& proto, const string& filename);

	bool ReadProtoFromBinaryFile(const string& filename, Message* proto);
	void WriteProtoToBinaryFile(const Message& proto, const string& filename);

	void ReadNetParamsFromTextFile(const string& param_file, NetParameter* param);
	void ReadNetParamsFromBinaryFile(const string& param_file, NetParameter* param);
	void ReadSolverParamsFromTextFile(const string& param_file, SolverParameter* param);

}  // namespace bromide

#endif
