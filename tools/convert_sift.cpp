#include <glog/logging.h>
#include <leveldb/db.h>
#include <leveldb/write_batch.h>
#include "matio.h"
#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <utility>

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/io.hpp"

using namespace caffe;
using std::pair;
using std::string;

const long int DATA_LEN = 31*31*128;
int edge[3] = {128,31,31};
int stride[3] = {1,1,1};
int start[3] = {0,0,0};


int read_mat_file(mat_t* matfp, float* data, float* output_data) {
	matvar_t *matvar;
	matvar = Mat_VarReadInfo(matfp, "sift_feature");
	if (Mat_VarReadData(matfp, matvar, data, start, stride, edge)){
		LOG(ERROR) << "read unsuccesfully!";
	}
	long int index = 0;
	long int out_index = 0;
	for (int c = 0; c < 128; c++) {
		for (int h = 0; h < 31; h++) {
			for (int w = 0; w < 31; w++) {
				index = 128*31*w + 128*h + c;
				output_data[out_index] = data[index];
				out_index ++;
			}
		}
	}
	Mat_VarFree(matvar);
	return EXIT_SUCCESS;
	
}

int main(int argc, char** argv) {
	::google::InitGoogleLogging(argv[0]);
	if (argc < 4) {
		printf("Convert a set of sift files to the leveldb format used\n"
			"as input for Caffe.\n"
			"Usage:\n"
			"    convert_imageset ROOTFOLDER/ LISTFILE DB_NAME"
			" RANDOM_SHUFFLE_DATA[0 or 1]\n"
			"The ImageNet dataset for the training demo is at\n"
			"    http://www.image-net.org/download-images\n");
		return 0;
	}
	
	std::ifstream infile(argv[2]);
	std::vector<std::pair<string, int> > lines;
	string filename;
	int label;
	string value;

	while(infile >> filename >> label) {
		lines.push_back(std::make_pair(filename, label));
	}
	
	if (argc == 5 && argv[4][0] == '1') {
		LOG(ERROR) << "Shuffling data";
		std::random_shuffle(lines.begin(), lines.end());
	}
	
	LOG(ERROR) << "A total of " << lines.size() << "images.";
	
	leveldb::DB* db;
	leveldb::Options options;
	options.error_if_exists = true;
	options.create_if_missing = true;
	options.write_buffer_size = 268435456;
	LOG(ERROR) << "Opening leveldb " << argv[3];
	
	leveldb::Status status = leveldb::DB::Open(options,argv[3], &db);
	CHECK(status.ok()) << "Failed to open leveldb: " << argv[3];
	
	string root_folder(argv[1]);
	Datum datum;
	datum.set_channels(128);
	datum.set_height(31);
	datum.set_width(31);
	for ( int idx = 0; idx < DATA_LEN; idx++) {
		datum.add_float_data(0);
	}
	int count = 0;
	const int kMaxKeyLength = 256;
	char key_cstr[kMaxKeyLength];
	leveldb::WriteBatch* batch = new leveldb::WriteBatch();
	float* data = new float[DATA_LEN]();
	float* out_data = new float[DATA_LEN]();
	for (int line_id = 0; line_id < lines.size(); ++line_id){
		mat_t *matfp;
                string mat_path = root_folder + lines[line_id].first;
		matfp = Mat_Open(mat_path.c_str(), MAT_ACC_RDONLY);
		if (NULL == matfp) {
			LOG(ERROR) << "Error opening .mat file " << root_folder + lines[line_id].first;
			return EXIT_FAILURE;
		}
		read_mat_file(matfp,data,out_data);
		Mat_Close(matfp);
		datum.set_label(lines[line_id].second);
		for (int idx = 0; idx < DATA_LEN; idx ++) {
			datum.set_float_data(idx, out_data[idx]);
		}
		datum.SerializeToString(&value);
		snprintf(key_cstr, kMaxKeyLength, "%08d_%s", line_id, lines[line_id].first.c_str());
		batch->Put(string(key_cstr), value);
		if (++count%1000 == 0) {
			db->Write(leveldb::WriteOptions(), batch);
			LOG(ERROR) << "Processed " << count << " files.";
			delete batch;
			batch = new leveldb::WriteBatch();
		}
		
	}
	if (count % 1000 !=0 ){
		db->Write(leveldb::WriteOptions(), batch);
		LOG(ERROR) << "Processed " << count << " files.";
	}
	delete batch;
	delete db;
	free(data);
	free(out_data);
	return 0;
	
	
	
}
