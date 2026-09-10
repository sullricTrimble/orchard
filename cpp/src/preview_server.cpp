#include "orchard/preview_server.hpp"
#include <opencv2/imgcodecs.hpp>
#ifdef _WIN32
namespace orchard {
PreviewServer::PreviewServer(int port) : port_(port) {}
PreviewServer::~PreviewServer() = default;
void PreviewServer::publish(const cv::Mat&) {}
void PreviewServer::run() {}
}
#else
#include <arpa/inet.h>
#include <ifaddrs.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <string>
namespace orchard {
static bool send_all(int fd, const char* p, size_t n) {
    while (n) {
        ssize_t w = ::send(fd, p, n, MSG_NOSIGNAL);
        if (w <= 0) return false;
        p += w; n -= (size_t)w;
    }
    return true;
}
static void print_urls(int port) {
    std::cout << "\n======== Guidance preview ========\n";
    std::cout << "No window appears in this terminal.\n";
    std::cout << "On your PC, open a browser at:\n";
    std::cout << "    http://127.0.0.1:" << port << "     (only if the browser is on the Pi)\n";
    ifaddrs* ifaddr = nullptr;
    if (getifaddrs(&ifaddr) == 0) {
        for (ifaddrs* p = ifaddr; p; p = p->ifa_next) {
            if (!p->ifa_addr || p->ifa_addr->sa_family != AF_INET) continue;
            char ip[INET_ADDRSTRLEN] = {};
            inet_ntop(AF_INET, &reinterpret_cast<sockaddr_in*>(p->ifa_addr)->sin_addr, ip, sizeof(ip));
            if (std::strcmp(ip, "127.0.0.1") == 0) continue;
            std::cout << "    http://" << ip << ":" << port << "\n";
        }
        freeifaddrs(ifaddr);
    }
    std::cout << "==================================\n\n";
    std::cout.flush();
}
PreviewServer::PreviewServer(int port) : port_(port) {
    if (port_ <= 0) return;
    listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) return;
    int yes = 1;
    ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons((uint16_t)port_);
    if (::bind(listen_fd_, (sockaddr*)&addr, sizeof(addr)) < 0) { ::close(listen_fd_); listen_fd_ = -1; return; }
    if (::listen(listen_fd_, 8) < 0) { ::close(listen_fd_); listen_fd_ = -1; return; }
    ok_ = true;
    print_urls(port_);
    th_ = std::thread([this]{ run(); });
}
PreviewServer::~PreviewServer() {
    stop_ = true;
    if (listen_fd_ >= 0) { ::shutdown(listen_fd_, SHUT_RDWR); ::close(listen_fd_); listen_fd_ = -1; }
    if (th_.joinable()) th_.join();
}
void PreviewServer::publish(const cv::Mat& bgr) {
    if (bgr.empty() || !ok_) return;
    std::vector<unsigned char> buf;
    cv::imencode(".jpg", bgr, buf, {cv::IMWRITE_JPEG_QUALITY, 70});
    std::lock_guard<std::mutex> lock(mu_);
    jpeg_.swap(buf);
}
void PreviewServer::serve_client(int client) {
    char req[1024];
    ssize_t n = ::recv(client, req, sizeof(req) - 1, 0);
    if (n <= 0) return;
    req[n] = 0;
    const bool video = std::strstr(req, "GET /video") != nullptr;
    if (!video) {
        const char* index =
            "HTTP/1.0 200 OK\r\nContent-Type: text/html\r\nConnection: close\r\n\r\n"
            "<!doctype html><html><head><meta charset='utf-8'><title>Row guidance</title>"
            "<style>html,body{margin:0;background:#050505;color:#eee;font-family:sans-serif}"
            "img{display:block;width:100%;height:100vh;object-fit:contain;background:#000}</style></head>"
            "<body><img src='/video' alt='guidance'></body></html>";
        send_all(client, index, std::strlen(index));
        return;
    }
    const char* head =
        "HTTP/1.0 200 OK\r\nCache-Control: no-cache, no-store\r\nPragma: no-cache\r\n"
        "Content-Type: multipart/x-mixed-replace; boundary=frame\r\n\r\n";
    if (!send_all(client, head, std::strlen(head))) return;
    while (!stop_) {
        std::vector<unsigned char> jpeg;
        {
            std::lock_guard<std::mutex> lock(mu_);
            jpeg = jpeg_;
        }
        if (jpeg.empty()) { ::usleep(40000); continue; }
        std::string part = "--frame\r\nContent-Type: image/jpeg\r\nContent-Length: "
            + std::to_string(jpeg.size()) + "\r\n\r\n";
        if (!send_all(client, part.data(), part.size())) break;
        if (!send_all(client, (const char*)jpeg.data(), jpeg.size())) break;
        if (!send_all(client, "\r\n", 2)) break;
        ::usleep(66000);
    }
}
void PreviewServer::run() {
    while (!stop_) {
        int client = ::accept(listen_fd_, nullptr, nullptr);
        if (client < 0) continue;
        std::thread([this, client]{
            serve_client(client);
            ::close(client);
        }).detach();
    }
}
}
#endif
