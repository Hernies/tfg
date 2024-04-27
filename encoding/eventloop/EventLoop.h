#include <iostream>
#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <tuple>
#include <vector>
#include <future>

extern "C" {
#include <mysql/mysql.h>
}

class EventLoop {
public:
    using AsyncFunction = std::function<void()>;
    using Condition = std::function<bool()>;
    using FulfillPromise = std::function<void()>;
    using Task = std::tuple<AsyncFunction, Condition, FulfillPromise>;

    EventLoop() : stopFlag(false) {
        loopThread = std::thread(&EventLoop::run, this);
    }

    ~EventLoop() {
        stop();
        loopThread.join();
    }

    void postTask(AsyncFunction asyncFunc, Condition condition, FulfillPromise fulfillPromise) {
        std::lock_guard<std::mutex> lock(queueMutex);
        tasksQueue.push(std::make_tuple(asyncFunc, condition, fulfillPromise));
        cv.notify_one();
    }

    void stop() {
        stopFlag = true;
        cv.notify_all();
    }

private:
    std::queue<Task> tasksQueue;
    std::vector<Task> waitingTasks;
    std::mutex queueMutex;
    std::condition_variable cv;
    std::atomic<bool> stopFlag;
    std::thread loopThread;

    void run() {
        while (!stopFlag) {
        if (stopFlag) break;
        // std::cout << "Executing task..." << std::endl;
        if (!tasksQueue.empty()) { 
            std::unique_lock<std::mutex> lock(queueMutex);
            cv.wait(lock, [this] { return !tasksQueue.empty() || stopFlag; });
            auto task = tasksQueue.front();
            tasksQueue.pop();
            lock.unlock();

            auto& [asyncFunc, condition, fulfillPromise] = task;
            asyncFunc();
            //check condition
            // std::cout << "Checking condition... "+ condition() << std::endl;
            if (!condition()) {
                // std::cout << "condition not met, pushing it back" << std::endl;
                waitingTasks.push_back(task);
            } else {
                // std::cout << "condition met " << std::endl;
                fulfillPromise();
            }
        }
        // std::cout << "Checking waiting tasks..." << std::endl;
        for (auto it = waitingTasks.begin(); it != waitingTasks.end(); ) {
            // std::cout << "loop" << std::endl;
            auto& [asyncFunc, condition, fulfillPromise] = *it;
                // print the contents of the vector
                // std::cout << "Checking condition... "+ condition() << std::endl;
            if (condition()) {
                // std::cout << "condition met " << std::endl;
                fulfillPromise();
                it = waitingTasks.erase(it);
            } else {
                // std::cout << "condition not met ";
                ++it;
            }
        // std::cout << "End of waiting tasks loop, starting over" << std::endl;
        }
        // std::cout << "End of event loop, starting over" << std::endl;
    }
    }
};