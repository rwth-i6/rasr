#ifndef _CORE_STATUS_HH
#define _CORE_STATUS_HH

#include <sstream>

namespace Core {

enum class StatusCode : int {
    Ok                 = 0,
    Cancelled          = 1,
    Unknown            = 2,
    InvalidArgument    = 3,
    DeadlineExceeded   = 4,
    NotFound           = 5,
    AlreadyExists      = 6,
    PermissionDenied   = 7,
    ResourceExhausted  = 8,
    FailedPrecondition = 9,
    Aborted            = 10,
    OutOfRange         = 11,
    Unimplemented      = 12,
    Internal           = 13,
    Unavailable        = 14,
    DataLoss           = 15,
    Unauthenticated    = 16,
    // RASR internal codes
    InvalidFileFormat = 100,
};

std::string statusCodeToString(StatusCode code);

class [[nodiscard]] Status final {
public:
    // This default constructor creates an OK status
    Status();

    // This constructor sets error message if not OK Status
    Status(StatusCode code, std::string msg);

    bool        ok() const;
    StatusCode  code() const;
    std::string message() const;

    void update(StatusCode code);
    void update(StatusCode code, std::string msg);
    void update(Status const& status);

private:
    StatusCode  code_;
    std::string msg_;
};

// inline implementations

inline bool Status::ok() const {
    return code_ == StatusCode::Ok;
}

inline StatusCode Status::code() const {
    return code_;
}

}  // namespace Core

#endif  // _CORE_STATUS_HH
