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
#include <Test/TokenizerFixture.hh>

namespace Tokenizer {

// Expected values are taken from the sentencepiece python package
class SentencePieceTokenizerTest : public Test::TokenizerFixture {
public:
    virtual void setUp() {
        TokenizerFixture::setUp();
        setTokenizerParameter("type", "sentencepiece");
        setTokenizerParameter("model-file", Test::dataFile("tokenizer/sentencepiece.model"));
    }
};

TEST_F(Tokenizer, SentencePieceTokenizerTest, Tokenize) {
    auto tokenizer = createTokenizer();
    auto tokens    = tokenize(*tokenizer, text);
    EXPECT_EQ(join(tokens), std::string("▁th e ▁new est ▁low er ▁tower ▁ i s ▁wide r ▁th an ▁ st r a ß e ▁ xy"));
    EXPECT_EQ(detokenize(*tokenizer, tokens), text);
}

TEST_F(Tokenizer, SentencePieceTokenizerTest, Vocabulary) {
    auto tokenizer = createTokenizer();
    EXPECT_EQ(tokenizer->vocabularySize(), 32u);
    EXPECT_EQ(tokenizer->tokenStrToId("<unk>"), 0u);
    EXPECT_EQ(tokenizer->tokenStrToId("▁tower"), 6u);
    EXPECT_EQ(tokenizer->tokenStrToId("xy"), AbstractTokenizer::invalidTokenId);
    EXPECT_EQ(tokenIdToStr(*tokenizer, 6u), std::string("▁tower"));
}

// Unknown pieces are encoded as the unknown id and decoded as " ⁇ "
TEST_F(Tokenizer, SentencePieceTokenizerTest, EncodeDecode) {
    auto tokenizer = createTokenizer();
    auto ids       = encode(*tokenizer, text);
    EXPECT_EQ(join(ids), std::string("3 4 15 18 10 8 6 5 12 9 17 14 3 16 5 7 14 20 27 4 5 0"));
    EXPECT_EQ(decode(*tokenizer, ids), std::string("the newest lower tower is wider than straße  ⁇ "));
}

TEST_F(Tokenizer, SentencePieceTokenizerTest, EmptyInput) {
    auto tokenizer = createTokenizer();
    EXPECT_TRUE(tokenize(*tokenizer, "").empty());
    EXPECT_TRUE(tokenize(*tokenizer, "   ").empty());
    EXPECT_TRUE(encode(*tokenizer, "").empty());
    EXPECT_EQ(detokenize(*tokenizer, {}), std::string());
    EXPECT_EQ(decode(*tokenizer, {}), std::string());
}

// With identity normalization only blanks become "▁", other whitespace is kept as characters
TEST_F(Tokenizer, SentencePieceTokenizerTest, Whitespace) {
    auto tokenizer = createTokenizer();
    auto tokens    = tokenize(*tokenizer, "the\ttower\nis  low");
    EXPECT_EQ(join(tokens), std::string("▁th e \t t o w er \n i s ▁low"));
    EXPECT_EQ(detokenize(*tokenizer, tokens), std::string("the\ttower\nis low"));
}

TEST_F(Tokenizer, SentencePieceTokenizerTest, Punctuation) {
    auto tokenizer = createTokenizer();
    auto tokens    = tokenize(*tokenizer, " lower. tower, (new)! ");
    EXPECT_EQ(join(tokens), std::string("▁low er . ▁tower , ▁ ( n e w )!"));
    EXPECT_EQ(detokenize(*tokenizer, tokens), std::string("lower. tower, (new)!"));
}

// Invalid UTF-8 bytes are replaced by U+FFFD
TEST_F(Tokenizer, SentencePieceTokenizerTest, RareCharacters) {
    auto tokenizer = createTokenizer();
    EXPECT_EQ(join(tokenize(*tokenizer, "🙂 tower")), std::string("▁ 🙂 ▁tower"));
    EXPECT_EQ(join(tokenize(*tokenizer, "\xff\xfelow")), std::string("▁ \uFFFD\uFFFD l o w"));
    EXPECT_EQ(join(encode(*tokenizer, "\xff\xfelow")), std::string("5 0 26 23 30"));
}

TEST_F(Tokenizer, SentencePieceTokenizerTest, InvalidTokenId) {
    auto tokenizer = createTokenizer();
    EXPECT_EQ(errorCode(tokenizer->tokenIdToStr(32u)), Core::StatusCode::OutOfRange);
    EXPECT_EQ(errorCode(tokenizer->decode({3u, 32u})), Core::StatusCode::OutOfRange);
    EXPECT_EQ(errorCode(tokenizer->wordBoundaryMarks(32u)), Core::StatusCode::OutOfRange);
}

TEST_F(Tokenizer, SentencePieceTokenizerTest, InvalidModel) {
    Test::Directory dir;
    setTokenizerParameter("model-file", "does-not-exist.model");
    EXPECT_EQ(errorCode(tryCreateTokenizer()), Core::StatusCode::NotFound);
    setTokenizerParameter("model-file", writeFile(dir, "malformed.model", "not a model"));
    EXPECT_EQ(errorCode(tryCreateTokenizer()), Core::StatusCode::InvalidFileFormat);
}

// Pieces that start a word begin with "▁", control pieces and the unknown piece are words of their own
TEST_F(Tokenizer, SentencePieceTokenizerTest, WordBoundaryMarks) {
    auto tokenizer = createTokenizer();
    EXPECT_EQ(formatWordBoundaryMarks(*tokenizer, encode(*tokenizer, "the lower tower")), std::string("▁th:B- e:-- ▁low:B- er:-- ▁tower:B-"));
    EXPECT_EQ(formatWordBoundaryMarks(*tokenizer, {0u, 1u, 2u}), std::string("<unk>:BA <s>:BA </s>:BA"));
}

// Models trained with `treat_whitespace_as_suffix` have the blank at the end of pieces instead
TEST_F(Tokenizer, SentencePieceTokenizerTest, WordBoundaryMarksSuffix) {
    setTokenizerParameter("model-file", Test::dataFile("tokenizer/sentencepiece_suffix.model"));
    auto tokenizer = createTokenizer();
    EXPECT_EQ(formatWordBoundaryMarks(*tokenizer, encode(*tokenizer, "wider newest")), std::string("w:-- i:-- d:-- er▁:-A newest▁:-A"));
    EXPECT_EQ(formatWordBoundaryMarks(*tokenizer, {0u, 1u}), std::string("<unk>:BA <s>:BA"));
}

// User-defined symbols are words of their own, also in id sequences that do not come from `encode`, e.g. a recognition
// result. `encode` puts a separate "▁" after them instead.
TEST_F(Tokenizer, SentencePieceTokenizerTest, WordBoundaryMarksUserDefinedSymbols) {
    setTokenizerParameter("model-file", Test::dataFile("tokenizer/sentencepiece_symbols.model"));
    auto                                    tokenizer = createTokenizer();
    std::vector<AbstractTokenizer::TokenId> ids;
    for (auto token : {"tower▁", "[NOISE]", "lower▁"}) {
        ids.push_back(tokenizer->tokenStrToId(token));
    }
    EXPECT_EQ(formatWordBoundaryMarks(*tokenizer, ids), std::string("tower▁:-A [NOISE]:BA lower▁:-A"));
}

}  // namespace Tokenizer
