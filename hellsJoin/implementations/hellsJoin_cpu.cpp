#include <iostream>
#include <fstream>
#include <sstream>
#include <atomic>
#include <vector>
#include <unordered_map>
#include <boost/bind.hpp>
#include <boost/thread.hpp>
#include <boost/thread/barrier.hpp>
#include <chrono>

#define START_M(name) \
    std::chrono::time_point<std::chrono::system_clock> start_name, end_name; \
    start_name = std::chrono::system_clock::now();

#define END_M(name) \
    end_name = std::chrono::system_clock::now(); \
    long elapsed_seconds_name = std::chrono::duration_cast<std::chrono::nanoseconds> (end_name-start_name).           count(); \
    std::cout << elapsed_seconds_name << "\n";
    //std::cout << name << " " << elapsed_seconds_name << " ns\n";

using std::vector;
using std::unordered_map;

struct record0 {
    int key;
    size_t ts;
    int value;
};

struct record1 {
    int key;
    size_t ts;
    int value;
};

struct record2 {
    int key;
    size_t ts;
    int left_value;
    int right_value;
};

using hash_map0 = unordered_map<int, vector<record0>>;
using hash_map1 = unordered_map<int, vector<record1>>;

std::atomic<size_t> meta0[2];
std::atomic<size_t> meta1[2];

hash_map0 hm0;
hash_map1 hm1;
vector<record2> result_stream;

boost::mutex m_build_and_probe;

void pipeline0(boost::barrier &bar, record0 *records, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        record0 record = records[i];

        //WINDOWING
        /*size_t ts = record.ts;
        for (auto &m:meta0) {
            size_t old = m;
            if (ts >= old) {
                size_t next = old + 2;
                if (m.compare_exchange_weak(old, next)) {
                    bar.wait();
                    hm0.clear();
                    bar.wait();
                }
            }
        }*/
        {
            // BUILD OWN HASH-TABLE
            //boost::lock_guard<boost::mutex> lock_guard(m_build_and_probe);
            if (hm0.find(record.key) == hm0.end()) {
                hm0[record.key] = vector<record0>();
            }
            hm0[record.key].push_back(record);

            // PROBE HASH-TABLE
            /*if (hm1.find(record.key) != hm1.end()) {
                for (auto &c : hm1[record.key]) {
                    record2 new_match = {record.key, record.ts, record.value, c.value};
                    result_stream.push_back(new_match);
                }
            }*/
        }
    }
}

void pipeline1(boost::barrier &bar, record1 *records, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        record1 record = records[i];

        //WINDOWING
        /*size_t ts = record.ts;
        for (auto &m:meta1) {
            size_t old = m;
            if (ts >= old) {
                size_t next = old + 2;
                if (m.compare_exchange_weak(old, next)) {
                    bar.wait();
                    hm1.clear();
                    bar.wait();
                }
            }
        }*/
        {
            // BUILD OWN HASH-TABLE
            /*boost::lock_guard<boost::mutex> lock_guard(m_build_and_probe);
            if (hm1.find(record.key) == hm1.end()) {
                hm1[record.key] = vector<record1>();
            }
            hm1[record.key].push_back(record);*/

            // PROBE HASH-TABLE
            if (hm0.find(record.key) != hm0.end()) {
                for (auto &c : hm0[record.key]) {
                    record2 new_match = {record.key, record.ts, c.value, record.value};
                    result_stream.push_back(new_match);
                }
            }
        }
    }
}

bool is_present(record2 cur) {
    for (auto &c : result_stream) {
        if (cur.key == c.key &&
            cur.ts == c.ts &&
            cur.left_value == c.left_value &&
            cur.right_value == c.right_value) {
            return true;
        }
    }
    return false;
}

void debug_print_result() {
    for (auto c : result_stream) {
        std::cout << "match  newtuple ("  << c.ts << ", " << c.key << ", "<< c.right_value << ", " << c.left_value << ")" << std::endl;
    }
}

void parseCSV(std::string filename, record0 *tup, int max){
    std::ifstream file(filename);
    std::string line;
    int row = 0;
    while (std::getline(file, line) && max > row){
        std::stringstream iss(line);
        std::string key, time, val;
        std::getline(iss, key , ',');
        std::getline(iss, time, ',');
        std::getline(iss, val , ',');

        tup[row] = {std::stoi(key), std::stoi(time), std::stoi(val)};
        row++;
    }
    std::cout << filename << ": " << row << "rows loaded" << std::endl;
};


void parseCSV(std::string filename, record1 *tup, int max){
    std::ifstream file(filename);
    std::string line;
    int row = 0;
    while (std::getline(file, line) && max > row){
        std::stringstream iss(line);
        std::string key, time, val;
        std::getline(iss, key , ',');
        std::getline(iss, time, ',');
        std::getline(iss, val , ',');

        tup[row] = {std::stoi(key), std::stoi(time), std::stoi(val)};
        row++;
    }
    std::cout << filename << ": " << row << "rows loaded" << std::endl;
};

void test_result(record2 * records_expected, size_t size_expected) {
    if (result_stream.size() != size_expected) {
        std::cout << "[Fail] Size of expected and actual result does not match!" << std::endl;
        exit(EXIT_FAILURE);
    }
    for (size_t i = 0; i < size_expected; ++i) {
        for (auto &c : result_stream) {
            if (!is_present(records_expected[i])) {
                std::cout << "[Fail] Different result!" << std::endl;
                exit(EXIT_FAILURE);
            }
        }
    }
}


int main(int argc, char *argv[]){
    if (argc != 5){
        printf("Usage: hellsjoin_file [filename1] [filename2] [rows] [window]");
    }

	int rows = atoi(argv[3]);
    int etpw = atoi(argv[4]);

	std::cout << "Buildsize: " << rows << std::endl;
	std::cout << "Probesize: " << etpw << std::endl;

	record0 *records0 = new record0[rows];
	record1 *records1 = new record1[etpw];
	parseCSV(argv[1], records0, rows);
	parseCSV(argv[2], records1, etpw);
	
	size_t ts = 1; // PoC is using ts in {1,2,3}
    meta0[0] = ts + 1;
    meta0[1] = ts + 2;
    meta1[0] = ts + 1;
    meta1[1] = ts + 2;

	int runs = 50;
	for (int i = 0; i< runs ; i++){
		boost::barrier bar(2);
		START_M(_)
		boost::thread thr0(boost::bind(&pipeline0, boost::ref(bar), records0, rows));
		thr0.join();
		boost::thread thr1(boost::bind(&pipeline1, boost::ref(bar), records1, etpw));
		thr1.join();
		END_M(_)
		hm0.clear();
		result_stream.clear();
	}
	hm0.clear();

    //debug_print_result();
    //test_result(records_expected, 11);

    return 0;
}
