#pragma once

#include <string>

namespace llm_infer {

class Status {
public:
  Status() = default;

  static Status ok() {
    return Status();
  }

  static Status error(std::string message) {
    Status status;
    status.ok_ = false;
    status.message_ = std::move(message);
    return status;
  }

  bool ok_status() const {
    return ok_;
  }

  const std::string& message() const {
    return message_;
  }

private:
  bool ok_ = true;
  std::string message_;
};

template <typename T>
class Result {
public:
  Result(T value) : value_(std::move(value)), status_(Status::ok()) {}
  Result(Status status) : status_(std::move(status)) {}

  bool ok() const {
    return status_.ok_status();
  }

  const Status& status() const {
    return status_;
  }

  const T& value() const {
    return value_;
  }

  T& value() {
    return value_;
  }

private:
  T value_{};
  Status status_;
};

}  // namespace llm_infer
