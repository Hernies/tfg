#include <iostream>
#include <sstream>
#include <fstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <vector>
#include <cppconn/driver.h>
#include <cppconn/connection.h>
#include <cppconn/statement.h>
#include <cppconn/resultset.h>
#include <mysql_driver.h>
using namespace std;



size_t countLines(char* mappedFile, size_t fileSize) {
    size_t lines = 0;
    for (size_t i = 0; i < fileSize; ++i) {
        if (mappedFile[i] == '\n') {
            ++lines;
        }
    }
    return lines;
}

void handleSQLException(const sql::SQLException &e, const string &file, const string &function, int line) {
    cout << "# ERR: SQLException in " << file;
    cout << "(" << function << ") on line " << line << endl;
    cout << "# ERR: " << e.what();
    cout << " (MySQL error code: " << e.getErrorCode();
    cout << ", SQLState: " << e.getSQLState() << " )" << endl;
}

char* insert(char*& currentPointer, char* fileEnd, const char* filepath, int house_id,std::unique_ptr<sql::Connection> &con) {
   
    std::vector<char> buffer;
    buffer.reserve(1024);
    //read a line from the file
    if (currentPointer >= fileEnd) {
        std::cout << "End of file reached." << std::endl;
        exit(0);
    }

    char* lineStart = currentPointer;
    while (currentPointer < fileEnd && *currentPointer != '\n') {
        ++currentPointer;
    }
    //save the line to a variable
    buffer.insert(buffer.end(), lineStart, currentPointer);
    //print the line
    std::cout.write(lineStart, currentPointer - lineStart);
    //now, split the line into fields each separated by a comma, and insert the fields into an array
    std::string line = std::string(buffer.begin(), buffer.end());
    //split the line into fields each separated by a comma
    std::stringstream ss(line);
    std::string item;
    std::vector<std::string> field;
    while (std::getline(ss, item, ',')) {
        field.push_back(item);
    }
    //print the contents of the row
    // prepare the insert statement
    std::string query = "INSERT INTO `datasets`.`datapoint` ( `House_idHouse`, `timestamp`, `aggregate`, `appliance1`, `appliance2`, `appliance3`, `appliance4`, `appliance5`, `appliance6`, `appliance7`, `appliance8`, `appliance9`, `issues`) ";
    query += "VALUES (" + std::to_string(house_id) + ", '" + field[0] + "', " + field[2] + ", " + field[3] + ", " + field[4] + ", " + field[5] + ", " + field[6] + ", " + field[7] + ", " + field[8] + ", " + field[9] + ", " + field[10] + ", " + field[11] + ", " + field[12] + ");";
    //print the query
    cout << query << endl;
    try{
        std::unique_ptr<sql::Statement> stmt(con->createStatement());
        stmt->execute(query);
        
    } catch (sql::SQLException &e) {
        handleSQLException(e, __FILE__, __FUNCTION__, __LINE__);
        exit (1);
    }
    // Move the pointer past the newline character for the next call
    if (currentPointer < fileEnd) {
        ++currentPointer;
    }
    return currentPointer;
}

void insertForever(char*& currentPointer, char* fileEnd, const char* filepath, int house_id,std::unique_ptr<sql::Connection> &con){
    
     std::vector<char> buffer;
    buffer.reserve(1024);
    //read a line from the file
    if (currentPointer >= fileEnd) {
        std::cout << "End of file reached." << std::endl;
        exit(0);
    }

    char* lineStart = currentPointer;
    while (currentPointer < fileEnd && *currentPointer != '\n') {
        ++currentPointer;
    }
    
    //print the contents of the row
    // prepare the insert statement
    int totalines = 0;
    std::string query = "INSERT INTO `datasets`.`datapoint` ( `House_idHouse`, `timestamp`, `aggregate`, `appliance1`, `appliance2`, `appliance3`, `appliance4`, `appliance5`, `appliance6`, `appliance7`, `appliance8`, `appliance9`, `issues`) VALUES\n";
    while (currentPointer < fileEnd) { 
        //save the line to a variable
        buffer.insert(buffer.end(), lineStart, currentPointer);
        //print the line
        std::cout.write(lineStart, currentPointer - lineStart);
        //now, split the line into fields each separated by a comma, and insert the fields into an array
        std::string line = std::string(buffer.begin(), buffer.end());
        //split the line into fields each separated by a comma
        std::stringstream ss(line);
        std::string item;
        std::vector<std::string> field;
        while (std::getline(ss, item, ',')) {
            field.push_back(item);
        }
        query += "(" + std::to_string(house_id) + ", '" + field[0] + "', " + field[2] + ", " + field[3] + ", " + field[4] + ", " + field[5] + ", " + field[6] + ", " + field[7] + ", " + field[8] + ", " + field[9] + ", " + field[10] + ", " + field[11] + ", " + field[12] + "),\n";
        
        if (currentPointer == fileEnd || totalines == 500) {
            //remove the last comma
            query.pop_back();
            query += ";";
            //print the query
            cout << query << endl;
            try{
                std::unique_ptr<sql::Statement> stmt(con->createStatement());
                stmt->execute(query);
                
            } catch (sql::SQLException &e) {
                handleSQLException(e, __FILE__, __FUNCTION__, __LINE__);
                std::cout << query << std::endl;
                exit (1);
            }
            query = "";
            query = "INSERT INTO `datasets`.`datapoint` ( `House_idHouse`, `timestamp`, `aggregate`, `appliance1`, `appliance2`, `appliance3`, `appliance4`, `appliance5`, `appliance6`, `appliance7`, `appliance8`, `appliance9`, `issues`) VALUES ";
            totalines = 0;
        }
        // Move the pointer past the newline character for the next call
        if (currentPointer < fileEnd) {
            ++currentPointer;
            ++totalines;
        } 

    }
}

void printNextLine(char*& currentPointer, char* fileEnd) {
    if (currentPointer >= fileEnd) {
        std::cout << "End of file reached." << std::endl;
        return;
    }

    char* lineStart = currentPointer;
    while (currentPointer < fileEnd && *currentPointer != '\n') {
        ++currentPointer;
    }

    // Print the line
    std::cout.write(lineStart, currentPointer - lineStart);
    std::cout << std::endl;

    // Move the pointer past the newline character for the next call
    if (currentPointer < fileEnd) {
        ++currentPointer;
    }
}
void prompt (char*& currentPointer,char*& mappedFile , char* fileEnd, const char* filepath, int house_id,std::unique_ptr<sql::Connection> &con){
    
    std::cout << "Press 'p' to print the next line \n ";
    std::cout << "'i' to continue inserting \n "; 
    std::cout << "'j' to jump to a specific line \n ";
    std::cout << "'r' to continue inserting until EOF\n";
    std::cout <<  "'s' to search for a line with a specific date\n";
    char command;
    std::cin >> command;
    switch(command){
        case 'p':
            printNextLine(currentPointer, fileEnd);
            break;
        case 'i':
            insert(currentPointer, fileEnd,filepath,house_id,con);
            break;
        case 'j':
            std::cout << "Enter the line number you want to jump to: ";
            size_t lineNumber;
            std::cin >> lineNumber;
            if (std::cin.fail() || lineNumber > countLines(mappedFile, fileEnd - mappedFile)) {
                std::cerr << "Invalid input or line number out of range." << std::endl;
                return;
            }
            for (size_t currentLine = 1; currentPointer < fileEnd && currentLine < lineNumber; ++currentPointer) {
                if (*currentPointer == '\n') {
                    ++currentLine;
                }
            }
            prompt (currentPointer, mappedFile,fileEnd,filepath,house_id,con);
            break;
        case 'r':
            insertForever(currentPointer, fileEnd,filepath,house_id,con);
            break;
        case 's':
            std::string date;
            std::cout << "Enter the date you want to search for: ";
            std::cin >> date;
            std::string dateToSearch = date + ",";
            char* found = std::search(currentPointer, fileEnd, dateToSearch.begin(), dateToSearch.end());
            if (found != fileEnd) {
                std::cout << "Date found at position: " << found - mappedFile << std::endl;
            } else {
                std::cout << "Date not found." << std::endl;
            }
            prompt(currentPointer, mappedFile,fileEnd,filepath,house_id,con);
            break;
    }

}
int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <file> <house_id>" << std::endl;
        return EXIT_FAILURE;
    }
    const char* filepath = argv[1];
    int house_id = atoi(argv[2]);
    int fd = open(filepath, O_RDONLY);
    if (fd == -1) {
        perror("Error opening file for reading");
        return EXIT_FAILURE;
    }

    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        perror("Error getting the file size");
        close(fd);
        return EXIT_FAILURE;
    }

    char* mappedFile = static_cast<char*>(mmap(nullptr, sb.st_size, PROT_READ, MAP_SHARED, fd, 0u));
    if (mappedFile == MAP_FAILED) {
        close(fd);
        perror("Error mapping the file");
        return EXIT_FAILURE;
    }
    char* currentPointer = mappedFile;
    char* fileEnd = mappedFile + sb.st_size;

    const string database = "datasets";
    std::unique_ptr<sql::Connection> con;
    try {
        sql::Driver* driver = get_driver_instance();
        con.reset(driver->connect("tcp://127.0.0.1:3306", "root", "root")); 
        con->setSchema(database);

        cout << "Successfully connected to the database!" << endl;
    } catch (sql::SQLException &e) {
        handleSQLException(e, __FILE__, __FUNCTION__, __LINE__);
        exit(1);
    }
    size_t totalLines = countLines(mappedFile, sb.st_size);
    std::cout << "Total lines in file: " << totalLines << std::endl;
    prompt(currentPointer, mappedFile,fileEnd,filepath,house_id,con);
    // Count the total number of lines


    munmap(mappedFile, sb.st_size);
    close(fd);
    return EXIT_SUCCESS;
}
