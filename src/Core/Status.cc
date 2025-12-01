/** Copyright 2025 RWTH Aachen University. All rights reserved.
 *
 *  Licensed under the RWTH ASR License (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.hltpr.rwth-aachen.de/rwth-asr/rwth-asr-license.html
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */
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
          msg_("") {};

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
