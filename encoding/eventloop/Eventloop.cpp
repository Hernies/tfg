#include "EventLoop.h"
#include <iostream>

EventLoop::EventLoop() : stopFlag(false) {
    loopThread = std::thread(&EventLoop::run, this);
}

EventLoop::~EventLoop() {
    stop();
    loopThread.join();
}

void EventLoop::postTask(AsyncFunction asyncFunc, Condition condition, FulfillPromise fulfillPromise) {
    std::lock_guard<std::mutex> lock(queueMutex);
    tasksQueue.push(std::make_tuple(asyncFunc, condition, fulfillPromise));
    cv.notify_one();
}

void EventLoop::stop() {
    stopFlag = true;
    cv.notify_all();
}

void EventLoop::run() {
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