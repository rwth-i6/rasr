#include "Status.hh"

namespace Core {
std::string statusCodeToString(StatusCode code) {
    switch (code) {
        case StatusCode::Ok:
            return "OK";
        case StatusCode::Cancelled:
            return "CANCELLED";
        case StatusCode::Unknown:
            return "UNKNOWN";
        case StatusCode::InvalidArgument:
            return "INVALID_ARGUMENT";
        case StatusCode::DeadlineExceeded:
            return "DEADLINE_EXCEEDED";
        case StatusCode::NotFound:
            return "NOT_FOUND";
        case StatusCode::AlreadyExists:
            return "ALREADY_EXISTS";
        case StatusCode::PermissionDenied:
            return "PERMISSION_DENIED";
        case StatusCode::ResourceExhausted:
            return "RESOURCE_EXHAUSTED";
        case StatusCode::FailedPrecondition:
            return "FAILED_PRECONDITION";
        case StatusCode::Aborted:
            return "ABORTED";
        case StatusCode::OutOfRange:
            return "OUT_OF_RANGE";
        case StatusCode::Unimplemented:
            return "UNIMPLEMENTED";
        case StatusCode::Internal:
            return "INTERNAL";
        case StatusCode::Unavailable:
            return "UNAVAILABLE";
        case StatusCode::DataLoss:
            return "DATA_LOSS";
        case StatusCode::Unauthenticated:
            return "UNAUTHENTICATED";
    }
    return "UNKNOWN_ERROR_TYPE";
}

Status::Status()
        : code_(StatusCode::Ok),
          msg_(""){};

Status::Status(StatusCode code, std::string msg)
        : code_(code),
          msg_("") {
    if (!ok()) {
        msg_ = msg;
    }
}

std::string Status::message() const {
    std::stringstream msg(statusCodeToString(code_));
    if (!msg_.empty()) {
        msg << ": " << msg_;
    }
    return msg.str();
}

void Status::update(StatusCode code) {
    update(code, "");
}

void Status::update(StatusCode code, std::string msg) {
    if (ok()) {
        code_ = code;
        msg_  = msg;
    }
}

void Status::update(Status const& status) {
    update(status.code(), status.message());
}

}  // namespace Core
