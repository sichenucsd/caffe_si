#include <stdint.h>
#include <leveldb/db.h>
#include <string>
#include <vector>

#include "caffe/util/io.hpp"

using std::string;

int main(int argc, char** argv) {

	leveldb::DB* db_temp;
	leveldb::Options options;
	options.create_if_missing = false;
	options.max_open_files = 100;

	string leveldb_name = "/deadline/chensi/imagenet2012/caffe_train/sift_train_leveldb";

	LOG(ERROR) << "Opening leveldb " << leveldb_name;

	leveldb::Status status = leveldb::DB::Open(options, leveldb_name, &db_temp);

	CHECK(status.ok()) << "Failed to open leveldb " << leveldb_name << std::endl << status.ToString();
	leveldb::Iterator* iter_;
	iter_=db_temp->NewIterator(leveldb::ReadOptions());
	iter_->SeekToFirst();

	caffe::Datum datum;

        datum.ParseFromString(iter_->value().ToString());
	//const string& data = datum.data();
	int datum_size = datum.channels()*datum.height()*datum.width();
	for (int i = 0; i <datum_size; i++) {
		LOG(ERROR) << datum.float_data(i);
                LOG(ERROR) << i;
	}
        LOG(ERROR) << iter_->key().ToString();
return 0;





}
