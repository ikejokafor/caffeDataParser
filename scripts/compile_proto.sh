protoc --cpp_out=./ caffe.proto
mv caffe.pb.h ../inc/
mv caffe.pb.cc ../src/
