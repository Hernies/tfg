#include <iostream>
#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <future>
#include <vector>

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
                std::cout << "Executing task..." << std::endl;
                task();
                lock.lock();
            }
        }
    }
};

int main() {
    EventLoop loop;

    // Example usage with std::promise and std::future
    std::promise<void> promise;
    auto future = promise.get_future();

    // Simulate an asynchronous operation
    loop.postTask([&promise]() {
        
        promise.set_value();
    });

    future.get();  // Wait for the operation to complete
    std::cout << "Operation completed." << std::endl;
    loop.stop();  // Stop the event loop

    return 0;
}
