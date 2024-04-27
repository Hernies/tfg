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
extern "C" {
#include <mysql/mysql.h>
}
#include "eventloop/EventLoop.h"
using namespace std;

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
};

//define a type called Series that stores a static array of DataPoint objects, 
//this array is determined by seriesSize YOU MUST CHANGE THIS TO A VALUE BEFORE COMPILING
#define SERIES_SIZE 256

struct Series {
    DataPoint dataPoints[SERIES_SIZE];
};

// the default values for the connection to the database are:
// static const char * c_host = "localhost";
// static const char * c_user = "root";
// static const char * c_auth = "root";
// static int          c_port = 3306;
// static const char * c_sock = NULL;
// static const char * c_dbnm = "datasets";

// the event loop that will be used to manage the threads' async tasks
EventLoop loop;

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
        int s=0;
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
                int appliance1 = row[4][0] != '\0';
                int appliance2 = row[5][0] != '\0';
                int appliance3 = row[6][0] != '\0';
                int appliance4 = row[7][0] != '\0';
                int appliance5 = row[8][0] != '\0';
                int appliance6 = row[9][0] != '\0';
                int appliance7 = row[10][0] != '\0';
                int appliance8 = row[11][0] != '\0';
                int appliance9 = row[12][0] != '\0';
                int issues = atoi(row[13]);

                //create a DataPoint object with the extracted values
                DataPoint dp = {iddatapoint, timestamp, aggregate, appliance1, appliance2, appliance3, appliance4, appliance5, appliance6, appliance7, appliance8, appliance9, issues};
                //if the series is full, print the series
                if (s == SERIES_SIZE-1) {
                    std::cout << "Series: " << std::endl;
                    s = 0;
                } else {
                    //add the DataPoint object to the series
                    Series series;
                    series.dataPoints[s] = dp;
                    s++;
                }
            }
        }
    }
}

void thGPUIO(){}

void inserter(MYSQL* conn){}   

int main(int argc, char* argv[]) {
    //the arguments for this program are: the house name, the batch size, the series size and the number of threads
    if (argc != 5) {
        cerr << "Usage: " << argv[0];
        cerr << " <house number> <number of threads> <batch size> ";
        cerr << "<number of batches>" << endl;
        cerr << "SERIES SIZE IS A DEFINE" << endl;
        return 1;
    }
    //initialise the mysql library befor spawning any threads 
    if (mysql_library_init(0, NULL, NULL)) {
        cerr << "mysql_library_init() failed" << endl;
        return 1;
    }
    
    //parse the arguments
    int house = atoi(argv[1]);
    int thAmmount = atoi(argv[2]);
    int batchSize = atoi(argv[3]);
    int times = atoi(argv[4]);

    // print all arguments



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

    for (auto &th : extractors) {
        th.join();
    }

}