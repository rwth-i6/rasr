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
    auto tokens    = tokenizer->tokenize(text);
    EXPECT_EQ(join(tokens), std::string("▁th e ▁new est ▁low er ▁tower ▁ i s ▁wide r ▁th an ▁ st r a ß e ▁ xy"));
    EXPECT_EQ(tokenizer->detokenize(tokens), text);
}

TEST_F(Tokenizer, SentencePieceTokenizerTest, Vocabulary) {
    auto tokenizer = createTokenizer();
    EXPECT_EQ(tokenizer->vocabularySize(), 32u);
    EXPECT_EQ(tokenizer->tokenId("<unk>"), 0u);
    EXPECT_EQ(tokenizer->tokenId("▁tower"), 6u);
    EXPECT_EQ(tokenizer->tokenId("xy"), AbstractTokenizer::invalidTokenId);
    EXPECT_EQ(tokenizer->token(6u), std::string("▁tower"));
}

// Unknown pieces are encoded as the unknown id and decoded as " ⁇ "
TEST_F(Tokenizer, SentencePieceTokenizerTest, EncodeDecode) {
    auto tokenizer = createTokenizer();
    auto ids       = tokenizer->encode(text);
    EXPECT_EQ(join(ids), std::string("3 4 15 18 10 8 6 5 12 9 17 14 3 16 5 7 14 20 27 4 5 0"));
    EXPECT_EQ(tokenizer->decode(ids), std::string("the newest lower tower is wider than straße  ⁇ "));
}

TEST_F(Tokenizer, SentencePieceTokenizerTest, EmptyInput) {
    auto tokenizer = createTokenizer();
    EXPECT_TRUE(tokenizer->tokenize("").empty());
    EXPECT_TRUE(tokenizer->tokenize("   ").empty());
    EXPECT_TRUE(tokenizer->encode("").empty());
    EXPECT_EQ(tokenizer->detokenize({}), std::string());
    EXPECT_EQ(tokenizer->decode({}), std::string());
}

// With identity normalization only blanks become "▁", other whitespace is kept as characters
TEST_F(Tokenizer, SentencePieceTokenizerTest, Whitespace) {
    auto tokenizer = createTokenizer();
    auto tokens    = tokenizer->tokenize("the\ttower\nis  low");
    EXPECT_EQ(join(tokens), std::string("▁th e \t t o w er \n i s ▁low"));
    EXPECT_EQ(tokenizer->detokenize(tokens), std::string("the\ttower\nis low"));
}

TEST_F(Tokenizer, SentencePieceTokenizerTest, Punctuation) {
    auto tokenizer = createTokenizer();
    auto tokens    = tokenizer->tokenize(" lower. tower, (new)! ");
    EXPECT_EQ(join(tokens), std::string("▁low er . ▁tower , ▁ ( n e w )!"));
    EXPECT_EQ(tokenizer->detokenize(tokens), std::string("lower. tower, (new)!"));
}

// Invalid UTF-8 bytes are replaced by U+FFFD
TEST_F(Tokenizer, SentencePieceTokenizerTest, RareCharacters) {
    auto tokenizer = createTokenizer();
    EXPECT_EQ(join(tokenizer->tokenize("🙂 tower")), std::string("▁ 🙂 ▁tower"));
    EXPECT_EQ(join(tokenizer->tokenize("\xff\xfelow")), std::string("▁ \uFFFD\uFFFD l o w"));
    EXPECT_EQ(join(tokenizer->encode("\xff\xfelow")), std::string("5 0 26 23 30"));
}

TEST_F(Tokenizer, SentencePieceTokenizerTest, InvalidTokenId) {
    delayErrors();
    auto tokenizer = createTokenizer();
    EXPECT_EQ(tokenizer->token(32u), std::string());
    EXPECT_TRUE(tokenizer->hasFatalErrors());
}

TEST_F(Tokenizer, SentencePieceTokenizerTest, DecodeInvalidTokenId) {
    delayErrors();
    auto tokenizer = createTokenizer();
    EXPECT_EQ(tokenizer->decode({3u, 32u}), std::string());
    EXPECT_TRUE(tokenizer->hasFatalErrors());
}

}  // namespace Tokenizer
