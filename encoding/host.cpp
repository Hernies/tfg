#include <iostream>
#include <sstream>
#include <fstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <vector>
#include <thread>
extern "C" {
#include <mysql/mysql.h>
}
using namespace std;


static const char * c_host = "localhost";
static const char * c_user = "root";
static const char * c_auth = "toot";
static int          c_port = 3306;
static const char * c_sock = "/usr/local/mysql/mysql.sock";
static const char * c_dbnm = "datasets";

void extractor (){
    //initiate connection to the database
    MYSQL_RES *result;
    MYSQL_ROW row;
    net_async_status status;
    MYSQL *conn = mysql_init(NULL);
    if (conn == NULL) {
        cerr << "mysql_init() failed" << endl;
        return;
    }
    if (mysql_real_connect_nonblocking(conn, c_host, c_user, c_auth, c_dbnm, c_port, c_sock, 0) == NULL) {
        cerr << "mysql_real_connect() failed" << endl;
        mysql_close(conn);
        return;
    }
    //query the database

}


int main(int argc, char* argv[]) {
    //the arguments for this program are: the house name, the batch size, the series size and the number of threads

    //initialise the mysql library befor spawning any threads 
    if (mysql_library_init(0, NULL, NULL)) {
        cerr << "mysql_library_init() failed" << endl;
        return 1;
    }
    std::thread threads[10];
    
    // Create and execute threads
    for (int i = 0; i < 10; ++i) {
        threads[i] = std::thread(extractor);
    }

    // Join threads
    for (auto &th : threads) {
        th.join();
    }

    return 0;

    

}