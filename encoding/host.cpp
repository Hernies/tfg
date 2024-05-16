#include <CL/cl2.hpp>
#include <iostream>
#include <sstream>
#include <fstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <vector>
#include <thread>
#include <future>
#include <string>
#include <cstring>
#include <algorithm>
extern "C" {
#include <mysql/mysql.h>
}
#include "thSafe/HashMap.h"
#include "thSafe/SafeQueue.hpp"
#include "eventloop/EventLoop.h"
using namespace std;


// the default values for the connection to the database are:
// static const char * c_host = "localhost";
// static const char * c_user = "root";
// static const char * c_auth = "root";
// static int          c_port = 3306;
// static const char * c_sock = NULL;
// static const char * c_dbnm = "datasets";



// the event loop that will be used to manage the threads' async tasks
EventLoop loop;

// datapoint struct
struct DataPoint {
    int iddatapoint;
    std::string timestamp;
    int aggregate;
    int appliance1;
    int appliance2;
    int appliance3;
    int appliance4;
    int appliance5;
    int appliance6;
    int appliance7;
    int appliance8;
    int appliance9;
    int issues;
    // the onehot vector to store the onehot encoding of the appliances
    std::vector<int> onehot = {appliance1!=0, appliance2!=0, appliance3!=0, appliance4!=0, appliance5!=0, appliance6!=0, appliance7!=0, appliance8!=0, appliance9!=0};
};

//define a type called Series that stores a static array of DataPoint objects, 
//this array is determined by seriesSize YOU MUST CHANGE THIS TO A VALUE BEFORE COMPILING
#define SERIES_SIZE 256

struct Series {
    DataPoint dataPoints[SERIES_SIZE];
    // the onehot vector to store the onehot encoding of the appliances for the series
    std::vector<int> onehot = {0, 0, 0, 0, 0, 0, 0, 0, 0};
    int min[10];
    int max[10];
};


//Thread safe data structures
CTSL::HashMap <int, Series> seriesMap;
SafeQueue<int> seriesQueue;

// the following functions are used to submit async tasks to the eventloop thread with the mysql library (currenty not fully implemented)

// get the connection to the database
MYSQL* getConn(MYSQL* connObject) {
    std::promise<MYSQL*> connectionPromise;

    auto asyncFunc = [connObject]() {
        mysql_real_connect_nonblocking(connObject, "localhost", "root", "root", "datasets", 3306, NULL, 0);
    };

    auto condition = [connObject]() -> bool {
        net_async_status status = mysql_real_connect_nonblocking(connObject, "localhost", "root", "root", "datasets", 3306, NULL, 0);
        return status == NET_ASYNC_COMPLETE || status == NET_ASYNC_ERROR;
    };

    auto fulfillPromise = [&connectionPromise, connObject]() {
        connectionPromise.set_value(connObject);

    };

    loop.postTask(asyncFunc, condition, fulfillPromise);
    // Wait for the connection to be established
    return connectionPromise.get_future().get();  // This will block until the connection is established or an error occurs
}

void execQuery(MYSQL* conn, std::string query) {
    std::promise<int> queryPromise;

    auto asyncFunc = [conn, query]() {
        mysql_real_query_nonblocking(conn, query.c_str(), query.size());
    };

    auto condition = [conn, query]() -> bool {
        net_async_status status = mysql_real_query_nonblocking(conn, query.c_str(), query.size());
        return status == NET_ASYNC_COMPLETE || status == NET_ASYNC_ERROR;
    };

    auto fulfillPromise = [&queryPromise, conn]() {
        queryPromise.set_value(1);
    };

    loop.postTask(asyncFunc, condition, fulfillPromise);
    // Wait for the connection to be established
    queryPromise.get_future().get();  // This will block until the connection is established or an error occurs
}

// void getResults(MYSQL* conn, MYSQL_RES** results) {
//     std::promise<bool> resultsPromise;

//     auto asyncFunc = [conn, &results]() {
//         mysql_store_result_nonblocking(conn, results);
//     };

//     auto condition = [conn, &results]() -> bool {
//         net_async_status status = mysql_store_result_nonblocking(conn, results);
//         return status == NET_ASYNC_COMPLETE || status == NET_ASYNC_ERROR;
//     };

//     auto fulfillPromise = [&resultsPromise, conn, results]() {
//         (*results!= nullptr)? resultsPromise.set_value(true) : resultsPromise.set_value(false);
//     };

//     loop.postTask(asyncFunc, condition, fulfillPromise);
//     //Wait for the connection to be established
//     (resultsPromise.get_future().get() )? std::cout<<"GOOD"<<std::endl : std::cout<< "BAD"<< std::endl;// This will block until the connection is established or an error occurs
//      (results!= nullptr)? std::cout<<"GOOD"<<std::endl : std::cout<< "BAD"<< std::endl;
// }

void getResults(MYSQL* conn, MYSQL_RES** results) {
    std::promise<bool> resultsPromise;
    // fixme results falls out of scope after lamda finishes
    // Since 'results' is captured by reference, there's no need to use '&results' in the function call.
    auto asyncFunc = [conn, &results]() {
        mysql_store_result_nonblocking(conn, results);
    };

    auto condition = [&resultsPromise, conn, &results]() -> bool {
        std::cout << "Before: " << results << " with value: " << (void*)*results << std::endl;
        net_async_status status = mysql_store_result_nonblocking(conn, results);
        std::cout << "After: " << results << " with value: " << (void*)*results << std::endl;
        std::cout << "Status: " << status << std::endl;
        if (status == NET_ASYNC_COMPLETE) {
            // Check for errors if result is NULL
            if (!*results) {
                return mysql_errno(conn) != 0 || mysql_field_count(conn) == 0;
            }
            return true;
        }
        return status == NET_ASYNC_ERROR;
    };

    auto fulfillPromise = [&resultsPromise, &results]() {
        std::cout << "fulfilling: " << results << " with value: " << (void*)*results << std::endl;
        resultsPromise.set_value(results != nullptr);
    };

    loop.postTask(asyncFunc, condition, fulfillPromise);

    // This will block until the operation is complete or an error occurs
    bool resultStatus = resultsPromise.get_future().get();
    std::cout << (resultStatus ? "GOOD" : "BAD") << std::endl;
    std::cout << (*results != nullptr ? "GOOD" : "BAD") << std::endl;
}


void extractor (MYSQL* connObject, int house, int batchSize, int times) {
    //async request to connect to database
    MYSQL * conn= nullptr;
    MYSQL_RES* results = nullptr;
    conn = getConn(connObject);
    if (conn == nullptr) {
        std::cerr << "Failed to connect to database" << std::endl;
        return;
    }

    for (int i = 0; i < times; ++i) {
        //calculate offset
        int offset = i * batchSize;
        //create the query
        std::string query = "SELECT * FROM datasets.datapoint WHERE House_idHouse=" + std::to_string(house) + " order by iddatapoint asc LIMIT " + std::to_string(batchSize) + " OFFSET " + std::to_string(offset);
        // std::cout << "Query: " << query << std::endl;
        // std::cout << std::to_string(mysql_real_query(conn, query.c_str(), query.length())) << std::endl;
        
        //execute the query
        mysql_real_query(conn, query.c_str(), query.length());
        results= mysql_store_result(conn);
        if (results == nullptr) {
            std::cerr << "Failed to get results" << std::endl;
            return;
        }

        // iterate over the rows
        Series series;
        int s=0;
        int min[10] = {0,0,0,0,0,0,0,0,0,0};
        int max[10] =  {0,0,0,0,0,0,0,0,0,0};
        for (int j = 0; j < batchSize; ++j) {
            MYSQL_ROW row;
            row = mysql_fetch_row(results);
            if (row == nullptr) {
                std::cerr << "Failed to fetch row" << std::endl;
                return;
            }
            else {
                // extract from the row the following values with the following types
                int iddatapoint = atoi(row[0]);
                std::string timestamp = row[2];
                int aggregate = atoi(row[3]);
                int appliance1 = atoi(row[4]);
                int appliance2 = atoi(row[5]);
                int appliance3 = atoi(row[6]);
                int appliance4 = atoi(row[7]);
                int appliance5 = atoi(row[8]);
                int appliance6 = atoi(row[9]);
                int appliance7 = atoi(row[10]);
                int appliance8 = atoi(row[11]);
                int appliance9 = atoi(row[12]);
                int issues = atoi(row[13]);

                //create a DataPoint object with the extracted values
                //if the series is full, print the series
                if (s == SERIES_SIZE-1) {
                    // std::cout << "Series: "<< s << std::endl;
                    //store the min and max values for the series for ease of scaling calculus in the gpu
                    
                    for (int i = 0; i < 10; ++i) {
                        series.min[i] = min[i];
                        series.max[i] = max[i];
                    }
                    

                    //store the series in the hash map

                    int key = iddatapoint-1;
                    seriesMap.insert(std::move(key), std::move(series));
                    //queue up the series to be processed by a gpuio into a concurrent queue
                    seriesQueue.Produce(std::move(key));
                    //reset the series
                    s = 0;
                } 
                DataPoint dp = {iddatapoint, timestamp, aggregate, appliance1, appliance2, appliance3, appliance4, appliance5, appliance6, appliance7, appliance8, appliance9, issues};
                //add the DataPoint object to the series
                series.dataPoints[s] = dp;
                //accumulate the max and min values for the aggregate and appliances
                min[0] = (aggregate < min[0])? aggregate : min[0];
                min[1] = (appliance1 < min[1])? appliance1 : min[1];
                min[2] = (appliance2 < min[2])? appliance2 : min[2];
                min[3] = (appliance3 < min[3])? appliance3 : min[3];
                min[4] = (appliance4 < min[4])? appliance4 : min[4];
                min[5] = (appliance5 < min[5])? appliance5 : min[5];
                min[6] = (appliance6 < min[6])? appliance6 : min[6];
                min[7] = (appliance7 < min[7])? appliance7 : min[7];
                min[8] = (appliance8 < min[8])? appliance8 : min[8];
                min[9] = (appliance9 < min[9])? appliance9 : min[9];
                
                max[0] = (aggregate > max[0])? aggregate : max[0];
                max[1] = (appliance1 > max[1])? appliance1 : max[1];
                max[2] = (appliance2 > max[2])? appliance2 : max[2];
                max[3] = (appliance3 > max[3])? appliance3 : max[3];
                max[4] = (appliance4 > max[4])? appliance4 : max[4];
                max[5] = (appliance5 > max[5])? appliance5 : max[5];
                max[6] = (appliance6 > max[6])? appliance6 : max[6];
                max[7] = (appliance7 > max[7])? appliance7 : max[7];
                max[8] = (appliance8 > max[8])? appliance8 : max[8];
                max[9] = (appliance9 > max[9])? appliance9 : max[9];

                
                // Accumulate the onehot vectors
                std::transform(series.onehot.begin(), series.onehot.end(), dp.onehot.begin(), series.onehot.begin(), std::plus<>());
                s++;
                
                //print the onehot vector for the series
                // std::cout << "Onehot: ";
                //    for (std::vector<int>::size_type i = 0; i < series.onehot.size(); ++i) {
                //         std::cout << series.onehot[i] << " ";
                //     }
                // std::cout << std::endl;
                
                
            }
        }
    }
}



void thGPUIO(){
    

    

    int patience = 0;
    int totalsubmitted = 0;
    int buffer=0;
    int numStructs = 20; // Assuming a value for numStructs
    int numVectors = 10; // Assuming a value for numVectors
    int vectorLength = 256; // Assuming a value for vectorLength
    Series series;
    //queue not empty
    while (true){
        //if the queue is empty sleep for a little bit
        if (seriesQueue.Size()<=0){
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            patience++;
            if (patience > 500){
                std::cout << "THATS IT, I HAVE NO MORE PATIENCE" << std::endl;
                break;
            }
            continue;
        }
        int key;
       
        

        if (buffer == numStructs-1){ //if the buffer is full, submit the buffer to the gpu
            // std::cout << "Buffer: " << buffer << std::endl;
            // Calculate the size of the buffer to be written
             
            //print on screen a message that the buffer has been read and the gaf matrix has been calculated
            // std::cout << "Buffer read, GAF matrix calculated" << std::endl;
            //print buffer contents to file, one file per series per struct
            for (int i = 0; i < numStructs; ++i) {
                std::ofstream file;
                std::string filename = "gaf" + std::to_string(i) + ".txt";
                file.open(filename);
                for (int j = 0; j < numVectors; ++j) {
                    for (int k = 0; k < vectorLength; ++k) {
                        
                    }
                    file << std::endl;
                }
                file.close();
            }
            //
            buffer = 0;
        }
}

void inserter(MYSQL* conn){}   


void setDBCache() {
     //initialise the mysql library befor spawning any threads 
    if (mysql_library_init(0, NULL, NULL)) {
        cerr << "mysql_library_init() failed" << endl;
        return;
    }
    //connect once to the database 
    MYSQL* conn = mysql_init(nullptr);
    if (!conn) {
        std::cerr << "Failed to initialize MYSQL object" << std::endl;
        return;
    }
    conn = getConn(conn);
    if (conn == nullptr) {
        std::cerr << "Failed to connect to database" << std::endl;
        return;
    }
    execQuery(conn, "SET GLOBAL innodb_buffer_pool_size = 8 * 1024 * 1024 * 1024;");
}
int main(int argc, char* argv[]) {
    //the arguments for this program are: the house name, the batch size, and the number of threads. 
    //The series size is specified by the define
    // std::cout << "argc: " << argc << std::endl;
    // for (int i = 0; i < argc; ++i) {
    //     std::cout << "argv[" << i << "]: " << argv[i] << std::endl;
    // }
    if (argc != 6) {
        cerr << "Usage: " << argv[0];
        cerr << " <house number> <number of threads> <batch size> ";
        cerr << "<number of batches>" << endl;
        cerr << "SERIES SIZE IS A DEFINE" << endl;
        return 1;
    }
    setDBCache();


    //parse the arguments
    int house = atoi(argv[1]);
    int thAmmount = atoi(argv[2]);
    int batchSize = atoi(argv[3]);
    int times = atoi(argv[4]);
int buffer=0;
    Series seriesBuffer[120];
    std::thread extractors[thAmmount];
    int threadBatchSize = batchSize / thAmmount;

    // Create and execute threads
    for (int i = 0; i < thAmmount; ++i) {
        MYSQL* conn = mysql_init(nullptr);
        if (!conn) {
            std::cerr << "Failed to initialize MYSQL object" << std::endl;
            return 1;
        }
        extractors[i] = std::thread(extractor, conn, house, threadBatchSize, times);
    }
    // initiate the gpuio thread
    std::thread gpuio(thGPUIO);
    
    //join the threads  

    for (auto &th : extractors) {
        th.join();
    }
    gpuio.join();

    //print the contents of the seriesMap and the seriesQueue
    // std::cout << "SeriesQuestd::thread gpuio(thGPUIO);ue size: " << seriesQueue.Size() << std::endl;
    
    //delete hashmaps    
    seriesMap.clear();

    //clean up the mysql library
    mysql_library_end();
    return 0;
}