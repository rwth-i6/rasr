/** Copyright 2026 RWTH Aachen University. All rights reserved.
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
#ifndef _CORE_EXPECTED_HH
#define _CORE_EXPECTED_HH

#include <expected>
#include <utility>

#include "Status.hh"

namespace Core {

/**
 * Return type of a function that produces a T or fails with a Status.
 *
 *   Expected<Pronunciation*> getPronunciation(std::string const& phon);
 *
 * Return a value as usual and an error as
 *   return std::unexpected(Status(StatusCode::InvalidArgument, "..."));
 *
 * Functions that produce no value return a plain Status instead.
 *
 * Unlike Status, std::expected is not [[nodiscard]], so the compiler does not
 * warn when the result of such a call is ignored.
 */
template<typename T>
using Expected = std::expected<T, Status>;

namespace internal {

/**
 * Error returned by TRY. Converts to the return type of the enclosing
 * function, which may be a Status or any Expected<U>.
 */
struct PropagatedError {
    Status status;

    operator Status() && {
        return std::move(status);
    }

    template<typename U>
    operator Expected<U>() && {
        return std::unexpected(std::move(status));
    }
};

inline bool isError(Status const& status) {
    return !status.ok();
}

template<typename T>
bool isError(Expected<T> const& expected) {
    return !expected.has_value();
}

inline Status takeError(Status&& status) {
    return std::move(status);
}

template<typename T>
Status takeError(Expected<T>&& expected) {
    return std::move(expected).error();
}

inline void takeValue(Status&&) {}

template<typename T>
T takeValue(Expected<T>&& expected) {
    return *std::move(expected);
}

}  // namespace internal

}  // namespace Core

/**
 * Evaluates expr, which must be a Status or an Expected<T>. On error, returns
 * the error from the enclosing function, whose return type must be a Status or
 * an Expected<U>. Otherwise TRY(expr) is the value of the Expected<T> (moved
 * out), or void for a Status:
 *
 *   auto phonemes = TRY(parsePronunciation(phon));
 *   TRY(checkFile(filename));
 *
 * An lvalue argument is copied, so TRY(x) leaves x unchanged; use
 * TRY(std::move(x)) for move-only values.
 *
 * Uses a GCC/Clang statement expression, so it can only appear inside a
 * function body (not in a member initializer list or default argument).
 */
#define TRY(expr)                                                                                         \
    ({                                                                                                    \
        auto _tryResult = (expr);                                                                         \
        if (::Core::internal::isError(_tryResult)) {                                                      \
            return ::Core::internal::PropagatedError{::Core::internal::takeError(std::move(_tryResult))}; \
        }                                                                                                 \
        ::Core::internal::takeValue(std::move(_tryResult));                                               \
    })

#endif  // _CORE_EXPECTED_HH
