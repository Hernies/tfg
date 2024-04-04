extern "C" {
#include <mysql/mysql.h>
}
#include <iostream>
#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <future>

class EventLoop {
public:
    using Task = std::function<void()>;

    EventLoop() : stopFlag(false) {
        loopThread = std::thread(&EventLoop::run, this);
    }

    ~EventLoop() {
        stop();
        loopThread.join();
    }

    // Post a new task to the event loop
    void postTask(const Task& task) {
        std::lock_guard<std::mutex> lock(queueMutex);
        tasksQueue.push(task);
        cv.notify_one();
    }

    // Stop the event loop
    void stop() {
        stopFlag = true;
        cv.notify_all();
    }

private:
    std::queue<Task> tasksQueue;
    std::mutex queueMutex;
    std::condition_variable cv;
    std::atomic<bool> stopFlag;
    std::thread loopThread;

    void run() {
        while (!stopFlag) {
            std::unique_lock<std::mutex> lock(queueMutex);
            cv.wait(lock, [this] { return !tasksQueue.empty() || stopFlag; });

            while (!tasksQueue.empty()) {
                auto task = tasksQueue.front();
                tasksQueue.pop();
                lock.unlock();
                task();
                lock.lock();
            }
        }
    }
};

int main() {
    EventLoop loop;

    static const char* c_host = "localhost";
    static const char* c_user = "root";
    static const char* c_auth = "root";
    static int c_port = 3306;
    static const char* c_dbnm = "datasets";

    std::promise<MYSQL*> promise;
    auto future = promise.get_future();

    // Post a task to connect to MySQL
    loop.postTask([&promise]() {
        MYSQL* conn = mysql_init(nullptr);
        if (!conn) {
            promise.set_exception(std::make_exception_ptr(std::runtime_error("Failed to initialize MYSQL object")));
            return;
        }
        
        if (!mysql_real_connect(conn, c_host, c_user, c_auth, c_dbnm, c_port, NULL, 0)) {
            mysql_close(conn);
            promise.set_exception(std::make_exception_ptr(std::runtime_error(mysql_error(conn))));
        } else {
            std::cout << "MySQL connection established." << std::endl;
            promise.set_value(conn);
        }
    });

    // Wait for the connection to be established
    try {
        MYSQL* conn = future.get();  // This will block until the connection is established or an error occurs
        // Use the connection...
        mysql_close(conn);
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    }

    loop.stop();  // Stop the event loop

    return 0;
}
