#include <sstream>
#include <fstream>
#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <vector>
#include <thread>
#include <future>
#include <string>
#include <cstring>
#include <math.h>
#include <algorithm>
extern "C" {
#include <mysql/mysql.h>
}
#include "thSafe/HashMap.h"
#include "thSafe/SafeQueue.hpp"
#include "eventloop/EventLoop.h"
#include <opencv2/opencv.hpp>
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
    std::vector<int> onehot = {appliance1!=0? 1:0, appliance2!=0? 1:0, appliance3!=0? 1:0, appliance4!=0? 1:0, appliance5!=0? 1:0, appliance6!=0? 1:0, appliance7!=0? 1:0, appliance8!=0? 1:0, appliance9!=0? 1:0};
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

struct SeriesMinMax {
    int min[10];
    int max[10];
};

// give me a matrix struct using vectors of vectors
struct Matrix {
    std::vector<std::vector<float>> matrix;
};


//Thread safe data structures
CTSL::HashMap <int, Series> seriesMap;
SafeQueue<int> seriesQueue;
SafeQueue<int> gafQueue;


 float scale_and_adjust(int value, int min, int max) {
    //return the scaled value of the series, only if it is inside the range [-1,1]
    //as an extra step, computes the acos of the value after it is scaled and adjusted
     float scaled;
    if (max == min) {
        scaled = 0;
    } else {
        scaled = (2.0f * value - (max + min)) / (max - min);
    }
    return (scaled > 1.0f) ? acosf(1.0f) : (scaled < -1.0f) ? acosf(-1.0f) : acosf(scaled);
}


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
                    
                    //store the min and max values for the series for ease of scaling calculus in the gpu
                    
                    for (int i = 0; i < 10; ++i) {
                        series.min[i] = min[i];
                        series.max[i] = max[i];
                    }
                    

                    //store the series in the hash map

                    int key = iddatapoint-1;
                    std::cout << "Inserting series: " << key << std::endl;
                    seriesMap.insert(std::move(key), std::move(series));
                    //queue up the series to be processed by a gpuio into a concurrent queue
                    seriesQueue.Produce(std::move(key));
                    //reset the series
                    series = Series();
                    s = 0;
                } 
                DataPoint dp = {iddatapoint, timestamp, aggregate, appliance1, appliance2, appliance3, appliance4, appliance5, appliance6, appliance7, appliance8, appliance9, issues};
                //add the DataPoint object to the series
                std::cout << "DataPoint: " << dp.onehot[0] << " " << dp.onehot[1] << " " << dp.onehot[2] << " " << dp.onehot[3] << " " << dp.onehot[4] << " " << dp.onehot[5] << " " << dp.onehot[6] << " " << dp.onehot[7] << " " << dp.onehot[8] << std::endl;
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

                
                // add the 1 or 0 of the datapoint onehot vector to the series onehot vector
                for (int i = 0; i < 9; ++i) {
                    series.onehot[i] += dp.onehot[i];
                }
                std::cout << "Series: " << series.onehot[0] << " " << series.onehot[1] << " " << series.onehot[2] << " " << series.onehot[3] << " " << series.onehot[4] << " " << series.onehot[5] << " " << series.onehot[6] << " " << series.onehot[7] << " " << series.onehot[8] << std::endl;
                // print the values of the onehot and the series onehot vector
               
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

auto apply_colormap = [](float value) -> cv::Vec3b {
    // Apply a simple rainbow colormap for demonstration
    float r = std::max(0.0f, 1.0f - std::fabs(2.0f * value - 1.0f));
    float g = 1.0f - std::fabs(2.0f * value - 1.0f);
    float b = std::max(0.0f, std::fabs(2.0f * value - 1.0f));

    return cv::Vec3b(static_cast<uchar>(b * 255), static_cast<uchar>(g * 255), static_cast<uchar>(r * 255));
};

void thGAF() {
    int patience = 0;
    int buffer = 0;
    int numStructs = 20; // Assuming a value for numStructs
    int numVectors = 10; // Assuming a value for numVectors
    int vectorLength = SERIES_SIZE; // Using SERIES_SIZE as vector length
    Series series;
    Series seriesBuffer[numStructs];
    int keyvalues[numStructs];


    // Queue not empty
    while (true) {
        int key;
        // If the queue is empty, sleep for a little bit
        if (seriesQueue.Size() <= 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
            patience++;
            if (patience > 1000) {
                std::cout << "THAT'S IT, I HAVE NO MORE PATIENCE" << std::endl;
                break;
            }
            continue;
        }

        if (buffer == numStructs) { // If the buffer is full, process the buffer
            // Write the buffer to a file, implementing the GAF encoding.
            for (int i = 0; i < numStructs; ++i) {
                
                std::ofstream file[numVectors];
                for (int j = 0; j < numVectors; ++j) {
                    //open file with name gaf+ the key value of the series

                    float scaledAggregate[vectorLength];
                    // float scaledAppliance1[vectorLength];
                    // float scaledAppliance2[vectorLength];
                    // float scaledAppliance3[vectorLength];
                    // float scaledAppliance4[vectorLength];
                    // float scaledAppliance5[vectorLength];
                    // float scaledAppliance6[vectorLength];
                    // float scaledAppliance7[vectorLength];
                    // float scaledAppliance8[vectorLength];
                    // float scaledAppliance9[vectorLength];



                    for (int k = 0; k < vectorLength; ++k) {
                        // Scale the series to the range [0,1] using the min and max values of the series
                        scaledAggregate[k] = scale_and_adjust(seriesBuffer[i].dataPoints[k].aggregate, seriesBuffer->min[0], seriesBuffer->max[0]);
                        // scaledAppliance1[k] = scale_and_adjust(seriesBuffer[i].dataPoints[k].appliance1, seriesBuffer->min[1], seriesBuffer->max[1]);
                        // scaledAppliance2[k] = scale_and_adjust(seriesBuffer[i].dataPoints[k].appliance2, seriesBuffer->min[2], seriesBuffer->max[2]);
                        // scaledAppliance3[k] = scale_and_adjust(seriesBuffer[i].dataPoints[k].appliance3, seriesBuffer->min[3], seriesBuffer->max[3]);
                        // scaledAppliance4[k] = scale_and_adjust(seriesBuffer[i].dataPoints[k].appliance4, seriesBuffer->min[4], seriesBuffer->max[4]);
                        // scaledAppliance5[k] = scale_and_adjust(seriesBuffer[i].dataPoints[k].appliance5, seriesBuffer->min[5], seriesBuffer->max[5]);
                        // scaledAppliance6[k] = scale_and_adjust(seriesBuffer[i].dataPoints[k].appliance6, seriesBuffer->min[6], seriesBuffer->max[6]);
                        // scaledAppliance7[k] = scale_and_adjust(seriesBuffer[i].dataPoints[k].appliance7, seriesBuffer->min[7], seriesBuffer->max[7]);
                        // scaledAppliance8[k] = scale_and_adjust(seriesBuffer[i].dataPoints[k].appliance8, seriesBuffer->min[8], seriesBuffer->max[8]);
                        // scaledAppliance9[k] = scale_and_adjust(seriesBuffer[i].dataPoints[k].appliance9, seriesBuffer->min[9], seriesBuffer->max[9]);
                    }
                    cv::Mat gafImage(vectorLength, vectorLength, CV_32F);
                    // cv::Mat gafImageAp1(vectorLength, vectorLength, CV_32F);
                    // cv::Mat gafImageAp2(vectorLength, vectorLength, CV_32F);
                    // cv::Mat gafImageAp3(vectorLength, vectorLength, CV_32F);
                    // cv::Mat gafImageAp4(vectorLength, vectorLength, CV_32F);
                    // cv::Mat gafImageAp5(vectorLength, vectorLength, CV_32F);
                    // cv::Mat gafImageAp6(vectorLength, vectorLength, CV_32F);
                    // cv::Mat gafImageAp7(vectorLength, vectorLength, CV_32F);
                    // cv::Mat gafImageAp8(vectorLength, vectorLength, CV_32F);
                    // cv::Mat gafImageAp9(vectorLength, vectorLength, CV_32F);

                    //open gaf text file

                    for (int k = 0; k < vectorLength; ++k) {
                        for (int l = 0; l < vectorLength; ++l) {
                            // Calculate the GAF encoding for the series
                            // Assuming gaf_encoding is a function that encodes based on k and l. the encoding is cos of the sum of the scaled values at positions k and l
                            float gafAggregate = cos(scaledAggregate[k] + scaledAggregate[l]);
                            // float gafAppliance1 = cos(scaledAppliance1[k] + scaledAppliance1[l]);
                            // float gafAppliance2 = cos(scaledAppliance2[k] + scaledAppliance2[l]);
                            // float gafAppliance3 = cos(scaledAppliance3[k] + scaledAppliance3[l]);
                            // float gafAppliance4 = cos(scaledAppliance4[k] + scaledAppliance4[l]);
                            // float gafAppliance5 = cos(scaledAppliance5[k] + scaledAppliance5[l]);
                            // float gafAppliance6 = cos(scaledAppliance6[k] + scaledAppliance6[l]);
                            // float gafAppliance7 = cos(scaledAppliance7[k] + scaledAppliance7[l]);
                            // float gafAppliance8 = cos(scaledAppliance8[k] + scaledAppliance8[l]);
                            // float gafAppliance9 = cos(scaledAppliance9[k] + scaledAppliance9[l]);
                            
                            // all images
                            gafImage.at<float>(k, l) = gafAggregate;
                            // gafImageAp1.at<float>(k, l) = gafAppliance1;
                            // gafImageAp2.at<float>(k, l) = gafAppliance2;
                            // gafImageAp3.at<float>(k, l) = gafAppliance3;
                            // gafImageAp4.at<float>(k, l) = gafAppliance4;
                            // gafImageAp5.at<float>(k, l) = gafAppliance5;
                            // gafImageAp6.at<float>(k, l) = gafAppliance6;
                            // gafImageAp7.at<float>(k, l) = gafAppliance7;
                            // gafImageAp8.at<float>(k, l) = gafAppliance8;
                            // gafImageAp9.at<float>(k, l) = gafAppliance9;
                        }
                    }

                    // Normalize the GAF image to [0, 1]
                    // Create a color image
                    cv::Mat colorImage(vectorLength, vectorLength, CV_8UC3);
                    // cv::Mat colorImageAp1(vectorLength, vectorLength, CV_8UC3);
                    // cv::Mat colorImageAp2(vectorLength, vectorLength, CV_8UC3);
                    // cv::Mat colorImageAp3(vectorLength, vectorLength, CV_8UC3);
                    // cv::Mat colorImageAp4(vectorLength, vectorLength, CV_8UC3);
                    // cv::Mat colorImageAp5(vectorLength, vectorLength, CV_8UC3);
                    // cv::Mat colorImageAp6(vectorLength, vectorLength, CV_8UC3);
                    // cv::Mat colorImageAp7(vectorLength, vectorLength, CV_8UC3);
                    // cv::Mat colorImageAp8(vectorLength, vectorLength, CV_8UC3);
                    // cv::Mat colorImageAp9(vectorLength, vectorLength, CV_8UC3);

                    for (int k = 0; k < vectorLength; ++k) {
                        for (int l = 0; l < vectorLength; ++l) {
                            colorImage.at<cv::Vec3b>(k, l) = apply_colormap(gafImage.at<float>(k, l));
                            // colorImageAp1.at<cv::Vec3b>(k, l) = apply_colormap(gafImageAp1.at<float>(k, l));
                            // colorImageAp2.at<cv::Vec3b>(k, l) = apply_colormap(gafImageAp2.at<float>(k, l));
                            // colorImageAp3.at<cv::Vec3b>(k, l) = apply_colormap(gafImageAp3.at<float>(k, l));
                            // colorImageAp4.at<cv::Vec3b>(k, l) = apply_colormap(gafImageAp4.at<float>(k, l));
                            // colorImageAp5.at<cv::Vec3b>(k, l) = apply_colormap(gafImageAp5.at<float>(k, l));
                            // colorImageAp6.at<cv::Vec3b>(k, l) = apply_colormap(gafImageAp6.at<float>(k, l));
                            // colorImageAp7.at<cv::Vec3b>(k, l) = apply_colormap(gafImageAp7.at<float>(k, l));
                            // colorImageAp8.at<cv::Vec3b>(k, l) = apply_colormap(gafImageAp8.at<float>(k, l));
                            // colorImageAp9.at<cv::Vec3b>(k, l) = apply_colormap(gafImageAp9.at<float>(k, l));
                        }
                    }

                    std::string filename = std::to_string(keyvalues[i]) + "_gaf" +".png";
                    cv::imwrite(filename, colorImage);
                    // std::string filenameAp1 = std::to_string(keyvalues[i]) + "_gafAp1" +".png";
                    // cv::imwrite(filenameAp1, colorImageAp1);
                    // std::string filenameAp2 = std::to_string(keyvalues[i]) + "_gafAp2" +".png";
                    // cv::imwrite(filenameAp2, colorImageAp2);
                    // std::string filenameAp3 = std::to_string(keyvalues[i]) + "_gafAp3" +".png";
                    // cv::imwrite(filenameAp3, colorImageAp3);
                    // std::string filenameAp4 = std::to_string(keyvalues[i]) + "_gafAp4" +".png";
                    // cv::imwrite(filenameAp4, colorImageAp4);
                    // std::string filenameAp5 = std::to_string(keyvalues[i]) + "_gafAp5" +".png";
                    // cv::imwrite(filenameAp5, colorImageAp5);
                    // std::string filenameAp6 = std::to_string(keyvalues[i]) + "_gafAp6" +".png";
                    // cv::imwrite(filenameAp6, colorImageAp6);
                    // std::string filenameAp7 = std::to_string(keyvalues[i]) + "_gafAp7" +".png";
                    // cv::imwrite(filenameAp7, colorImageAp7);
                    // std::string filenameAp8 = std::to_string(keyvalues[i]) + "_gafAp8" +".png";
                    // cv::imwrite(filenameAp8, colorImageAp8);
                    // std::string filenameAp9 = std::to_string(keyvalues[i]) + "_gafAp9" +".png";
                    // cv::imwrite(filenameAp9, colorImageAp9);

                    //insert into the queue the key value of the series
                    gafQueue.Produce(std::move(keyvalues[i]));

                }
            }
            buffer = 0;
        }

        if (!seriesQueue.ConsumeSync(key)) {
            std::cerr << "Failed to consume key from seriesQueue" << std::endl;
            continue;
        }

        // Get the series from the hashmap
        if (!seriesMap.find(key, series)) {
            std::cerr << "Failed to find series in seriesMap" << std::endl;
            continue;
        }

        // Store the series in the buffer
        keyvalues[buffer] = key;
        seriesBuffer[buffer] = series;

        buffer++;
    }
}

void metadatos(){
// connect to database
    
    int buffer = 0;
    int numStructs = 20; // Assuming a value for numStructs
    Series series;
    Series seriesBuffer[numStructs];
    int keyvalues[numStructs];

    // Queue not empty
    while (true) {
        int key;
        // If the queue is empty, sleep for a little bit
        if (gafQueue.Size() <= 0) {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            
            continue;
        }

        if (buffer == numStructs) { // If the buffer is full, insert the metadata into the database
            for (int i = 0; i < numStructs; ++i) {
                // write the onehotvector to a file with the key value of the series
                std::ofstream file;
                std::string filename =std::to_string(keyvalues[i]) +  "_onehot" + ".txt";
                file.open(filename);
                file << std::to_string(keyvalues[i])+ ",";
                for (int j = 0; j < 9; ++j) {
                    file << seriesBuffer[i].onehot[j] << ",";
                }
                file << std::endl;
            }
            buffer = 0;
        }

        if (!gafQueue.ConsumeSync(key)) {
            std::cerr << "Failed to consume key from seriesQueue" << std::endl;
            continue;
        }

        // Get the series from the hashmap
        if (!seriesMap.find(key, series)) {
            std::cerr << "Failed to find series in seriesMap" << std::endl;
            continue;
        }

         // Store the series in the buffer
        keyvalues[buffer] = key;
        seriesBuffer[buffer] = series;

        buffer++;
    }
}   


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
    execQuery(conn, "SET GLOBAL innodb_buffer_pool_instances = 8;");
    //close the connection
    mysql_close(conn);

    
}

int main(int argc, char* argv[]) {
    //the arguments for this program are: the house name, the batch size, and the number of threads. 
    //The series size is specified by the define
    // std::cout << "argc: " << argc << std::endl;
    // for (int i = 0; i < argc; ++i) {
    //     std::cout << "argv[" << i << "]: " << argv[i] << std::endl;
    // }
    if (argc != 5) {
        cerr << "Usage: " << argv[0];
        cerr << " <house number> <number of threads> <batch size> ";
        cerr << "<number of batches>" << endl;
        cerr << "SERIES SIZE IS A DEFINE" << endl;
        return 1;
    }
    setDBCache();


    //parse the arguments
    int house = atoi(argv[1]);
    int thAmmount = atoi(argv[2])*3;
    int batchSize = atoi(argv[3]);
    int times = atoi(argv[4]);
    Series seriesBuffer[120];
    std::thread threads[thAmmount];
    int threadBatchSize = batchSize / thAmmount;

    // Create and execute threads
    int i = 0;
    while ( i < thAmmount ) {
        MYSQL* conn = mysql_init(nullptr);
        if (!conn) {
            std::cerr << "Failed to initialize MYSQL object" << std::endl;
            return 1;
        }
        threads[i] = std::thread(extractor, conn, house, threadBatchSize, times);
        ++i;
        std::cout << "Starting GAF thread" << std::endl;
        threads[i] = std::thread(thGAF); 
        ++i;
        std::cout << "Starting metadata thread" << std::endl;
        threads[i] = std::thread(metadatos);
        ++i;
    }
    // initiate the GAF thread

    
    //join the threads  

    for (auto &th : threads) {
        th.join();
    }


    //print the contents of the seriesMap and the seriesQueue
    // std::cout << "SeriesQuestd::thread gaf(thGAF);ue size: " << seriesQueue.Size() << std::endl;
    
    //delete hashmaps    
    seriesMap.clear();

    //clean up the mysql library
    mysql_library_end();
    return 0;
}