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
#include <memory>
#include <string>

#include <Core/Expected.hh>
#include <Test/UnitTest.hh>

namespace Core {

namespace {

Expected<int> parseNumber(std::string const& s) {
    if (s == "one") {
        return 1;
    }
    return std::unexpected(Status(StatusCode::InvalidArgument, "not a number: " + s));
}

Status checkPositive(int value) {
    if (value <= 0) {
        return Status(StatusCode::OutOfRange, "not positive");
    }
    return Status();
}

Expected<std::unique_ptr<int>> makeBoxed(bool fail) {
    if (fail) {
        return std::unexpected(Status(StatusCode::Internal, "no box"));
    }
    return std::make_unique<int>(7);
}

// Counts how often the code after a TRY was reached.
int reachedAfterTry = 0;

Expected<int> addOne(std::string const& s) {
    int value = TRY(parseNumber(s));
    ++reachedAfterTry;
    return value + 1;
}

Expected<int> checkThenReturn(int value) {
    TRY(checkPositive(value));
    ++reachedAfterTry;
    return value;
}

Status parseIntoStatus(std::string const& s) {
    int value = TRY(parseNumber(s));
    ++reachedAfterTry;
    return checkPositive(value);
}

Status checkIntoStatus(int value) {
    TRY(checkPositive(value));
    ++reachedAfterTry;
    return Status();
}

Expected<int> unbox(bool fail) {
    std::unique_ptr<int> boxed = TRY(makeBoxed(fail));
    return *boxed;
}

}  // namespace

TEST(Core, Expected, TryUnwrapsValue) {
    reachedAfterTry      = 0;
    Expected<int> result = addOne("one");
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(2, *result);
    EXPECT_EQ(1, reachedAfterTry);
}

TEST(Core, Expected, TryReturnsErrorOfExpected) {
    reachedAfterTry      = 0;
    Expected<int> result = addOne("two");
    EXPECT_FALSE(result.has_value());
    EXPECT_TRUE(result.error().code() == StatusCode::InvalidArgument);
    EXPECT_EQ(std::string("INVALID_ARGUMENT: not a number: two"), result.error().message());
    EXPECT_EQ(0, reachedAfterTry);
}

TEST(Core, Expected, TryOnOkStatusContinues) {
    reachedAfterTry      = 0;
    Expected<int> result = checkThenReturn(3);
    EXPECT_TRUE(result.has_value());
    EXPECT_EQ(3, *result);
    EXPECT_EQ(1, reachedAfterTry);
}

TEST(Core, Expected, TryReturnsErrorOfStatus) {
    reachedAfterTry      = 0;
    Expected<int> result = checkThenReturn(0);
    EXPECT_FALSE(result.has_value());
    EXPECT_TRUE(result.error().code() == StatusCode::OutOfRange);
    EXPECT_EQ(0, reachedAfterTry);
}

TEST(Core, Expected, TryInStatusFunction) {
    reachedAfterTry = 0;
    EXPECT_TRUE(parseIntoStatus("one").ok());
    EXPECT_TRUE(checkIntoStatus(1).ok());
    EXPECT_EQ(2, reachedAfterTry);

    reachedAfterTry     = 0;
    Status fromExpected = parseIntoStatus("two");
    EXPECT_TRUE(fromExpected.code() == StatusCode::InvalidArgument);
    EXPECT_EQ(std::string("INVALID_ARGUMENT: not a number: two"), fromExpected.message());
    Status fromStatus = checkIntoStatus(0);
    EXPECT_TRUE(fromStatus.code() == StatusCode::OutOfRange);
    EXPECT_EQ(0, reachedAfterTry);
}

TEST(Core, Expected, TryMovesValueOut) {
    Expected<int> ok = unbox(false);
    EXPECT_TRUE(ok.has_value());
    EXPECT_EQ(7, *ok);

    Expected<int> failed = unbox(true);
    EXPECT_FALSE(failed.has_value());
    EXPECT_TRUE(failed.error().code() == StatusCode::Internal);
}

TEST(Core, Expected, TryLeavesNamedExpectedIntact) {
    Expected<std::string> name = std::string("lexicon");
    auto                  copy = [&]() -> Expected<std::string> {
        return TRY(name);
    };
    EXPECT_EQ(std::string("lexicon"), *copy());
    EXPECT_EQ(std::string("lexicon"), *name);
}

}  // namespace Core
