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
#include <Core/Status.hh>
#include <Test/UnitTest.hh>

namespace Core {

TEST(Core, Status, OkMessage) {
    EXPECT_EQ(std::string("OK"), Status().message());
}

TEST(Core, Status, MessageContainsCodeAndText) {
    EXPECT_EQ(std::string("NOT_FOUND: x"), Status(StatusCode::NotFound, "x").message());
    EXPECT_EQ(std::string("INVALID_ARGUMENT: Unknown phoneme: \"zz\""), Status(StatusCode::InvalidArgument, "Unknown phoneme: \"zz\"").message());
}

TEST(Core, Status, MessageWithoutText) {
    EXPECT_EQ(std::string("INTERNAL"), Status(StatusCode::Internal, "").message());
}

TEST(Core, Status, InvalidFileFormatHasName) {
    EXPECT_EQ(std::string("INVALID_FILE_FORMAT: bad"), Status(StatusCode::InvalidFileFormat, "bad").message());
}

TEST(Core, Status, UpdateFromStatusKeepsMessage) {
    Status error(StatusCode::NotFound, "x");
    Status status;
    status.update(error);
    EXPECT_TRUE(status.code() == StatusCode::NotFound);
    EXPECT_EQ(std::string("NOT_FOUND: x"), status.message());
}

TEST(Core, Status, UpdateKeepsFirstError) {
    Status status(StatusCode::NotFound, "first");
    status.update(Status(StatusCode::Internal, "second"));
    EXPECT_TRUE(status.code() == StatusCode::NotFound);
    EXPECT_EQ(std::string("NOT_FOUND: first"), status.message());
}

}  // namespace Core
