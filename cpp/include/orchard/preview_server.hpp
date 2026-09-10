#pragma once
#include <opencv2/core.hpp>
#include <atomic>
#include <mutex>
#include <thread>
#include <vector>
namespace orchard {
class PreviewServer {
public:
    explicit PreviewServer(int port);
    ~PreviewServer();
    PreviewServer(const PreviewServer&) = delete;
    PreviewServer& operator=(const PreviewServer&) = delete;
    void publish(const cv::Mat& bgr);
    bool ok() const { return ok_; }
    int port() const { return port_; }
private:
    void run();
    void serve_client(int client);
    int port_ = 0;
    int listen_fd_ = -1;
    std::atomic<bool> ok_{false};
    std::atomic<bool> stop_{false};
    std::mutex mu_;
    std::vector<unsigned char> jpeg_;
    std::thread th_;
};
}
