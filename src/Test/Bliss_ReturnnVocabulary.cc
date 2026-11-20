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
#include <fstream>
#include <stdexcept>

#include <Bliss/ReturnnVocabulary.hh>
#include <Test/File.hh>
#include <Test/UnitTest.hh>

namespace Bliss {

namespace {

// Expected values are taken from `returnn.datasets.util.vocabulary.Vocabulary`
Core::Status loadContent(std::string const& content, ReturnnVocabulary& vocabulary) {
    Test::Directory dir;
    std::string     path = Test::File(dir, "vocab.py").path();
    std::ofstream(path.c_str()) << content;
    return ReturnnVocabulary::load(path, vocabulary);
}

}  // namespace

// As written by create-py-vocab.py of rwth-i6/subword-nmt
TEST(Bliss, ReturnnVocabulary, Load) {
    ReturnnVocabulary vocabulary;
    Core::Status      status = ReturnnVocabulary::load(::Test::dataFile("tokenizer/bpe.vocab.py"), vocabulary);
    CPPUNIT_ASSERT_MESSAGE(status.message(), status.ok());
    EXPECT_EQ(vocabulary.size(), 34u);
    EXPECT_EQ(vocabulary.id("tower"), 3u);
    EXPECT_EQ(vocabulary.token(3u), std::string("tower"));
    EXPECT_EQ(vocabulary.id("x@@"), ReturnnVocabulary::invalidId);
}

// Of several tokens with the same id, the one that sorts last is used for the id
TEST(Bliss, ReturnnVocabulary, SharedIds) {
    ReturnnVocabulary vocabulary;
    EXPECT_TRUE(loadContent("{'</s>': 0, '<s>': 0, 'a': 1, 'c': 1, 'b': 1}", vocabulary).ok());
    EXPECT_EQ(vocabulary.size(), 2u);
    EXPECT_EQ(vocabulary.id("</s>"), 0u);
    EXPECT_EQ(vocabulary.id("a"), 1u);
    EXPECT_EQ(vocabulary.token(0u), std::string("<s>"));
    EXPECT_EQ(vocabulary.token(1u), std::string("c"));
}

// As in a python dict literal, the last entry of a token counts
TEST(Bliss, ReturnnVocabulary, DuplicateTokens) {
    ReturnnVocabulary vocabulary;
    EXPECT_TRUE(loadContent("{'a': 0, 'b': 0, 'a': 1}", vocabulary).ok());
    EXPECT_EQ(vocabulary.id("a"), 1u);
    EXPECT_EQ(vocabulary.token(0u), std::string("b"));
    EXPECT_EQ(vocabulary.token(1u), std::string("a"));
}

// Quotes and escape sequences as written by python's `repr`
TEST(Bliss, ReturnnVocabulary, Escapes) {
    ReturnnVocabulary vocabulary;
    EXPECT_TRUE(loadContent("{'<s>': 0, \"it's\": 1, 'a\\'b\\\\': 2, '\\u200b@@': 3, '\\xad': 4, 'ß': 5, '\\U0001f642': 6}\n", vocabulary).ok());
    EXPECT_EQ(vocabulary.size(), 7u);
    EXPECT_EQ(vocabulary.id("it's"), 1u);
    EXPECT_EQ(vocabulary.id("a'b\\"), 2u);
    EXPECT_EQ(vocabulary.id("​@@"), 3u);
    EXPECT_EQ(vocabulary.id("­"), 4u);
    EXPECT_EQ(vocabulary.id("ß"), 5u);
    EXPECT_EQ(vocabulary.id("🙂"), 6u);
}

TEST(Bliss, ReturnnVocabulary, MissingFile) {
    ReturnnVocabulary vocabulary;
    EXPECT_EQ(ReturnnVocabulary::load("does-not-exist.py", vocabulary).code(), Core::StatusCode::NotFound);
}

// Including unused ids, which RETURNN rejects as well
TEST(Bliss, ReturnnVocabulary, Malformed) {
    ReturnnVocabulary vocabulary;
    for (std::string content : {"", "{}", "['a']", "{'a': 0", "{'a' 0}", "{'a': -1}", "{'a': 1}", "{'a': 0, 'b': 2}", "{'a': 0, 'b': 1, 'a': 2}", "{'\\x4': 0}"}) {
        EXPECT_EQ(loadContent(content, vocabulary).code(), Core::StatusCode::InvalidFileFormat);
    }
}

TEST(Bliss, ReturnnVocabulary, InvalidId) {
    ReturnnVocabulary vocabulary;
    EXPECT_TRUE(loadContent("{'a': 0}", vocabulary).ok());
    CPPUNIT_ASSERT_THROW(vocabulary.token(1u), std::out_of_range);
}

}  // namespace Bliss
