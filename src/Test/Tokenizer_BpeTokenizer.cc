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

#include <Test/TokenizerFixture.hh>

namespace Tokenizer {

// Expected values are taken from subword-nmt 0.3.8 (apply_bpe.py)
class BpeTokenizerTest : public Test::TokenizerFixture {
public:
    virtual void setUp() {
        TokenizerFixture::setUp();
        setTokenizerParameter("type", "bpe");
        setTokenizerParameter("codes-file", Test::dataFile("tokenizer/bpe.codes"));
    }

protected:
    std::string writeFile(Test::Directory const& dir, std::string const& name, std::string const& content) {
        std::string   path = Test::File(dir, name).path();
        std::ofstream os(path.c_str());
        os << content;
        return path;
    }
};

TEST_F(Tokenizer, BpeTokenizerTest, Tokenize) {
    auto tokenizer = createTokenizer();
    auto tokens    = tokenizer->tokenize(text);
    EXPECT_EQ(join(tokens), std::string("the newest lower tower is wide@@ r than s@@ t@@ r@@ a@@ ß@@ e x@@ y"));
    EXPECT_EQ(tokenizer->detokenize(tokens), text);
    EXPECT_EQ(join(tokenizer->tokenize(" the \t newest  ")), std::string("the newest"));
}

TEST_F(Tokenizer, BpeTokenizerTest, EmptyInput) {
    auto tokenizer = createTokenizer();
    EXPECT_TRUE(tokenizer->tokenize("").empty());
    EXPECT_TRUE(tokenizer->tokenize(" \t\n ").empty());
    EXPECT_EQ(tokenizer->detokenize({}), std::string());
}

// All whitespace characters are word boundaries and are normalized to single blanks
TEST_F(Tokenizer, BpeTokenizerTest, Whitespace) {
    auto tokenizer = createTokenizer();
    auto tokens    = tokenizer->tokenize("the\ttower\nis  low\r\n");
    EXPECT_EQ(join(tokens), std::string("the tower is l@@ o@@ w"));
    EXPECT_EQ(tokenizer->detokenize(tokens), std::string("the tower is low"));
}

TEST_F(Tokenizer, BpeTokenizerTest, Punctuation) {
    auto tokenizer = createTokenizer();
    auto tokens    = tokenizer->tokenize("lower. tower, (new)!");
    EXPECT_EQ(join(tokens), std::string("l@@ owe@@ r@@ . t@@ owe@@ r@@ , (@@ ne@@ w@@ )@@ !"));
    EXPECT_EQ(tokenizer->detokenize(tokens), std::string("lower. tower, (new)!"));
}

// Multi-byte characters are never split, invalid UTF-8 is passed through byte-wise
TEST_F(Tokenizer, BpeTokenizerTest, RareCharacters) {
    auto tokenizer = createTokenizer();
    EXPECT_EQ(join(tokenizer->tokenize("🙂 tower")), std::string("🙂 tower"));
    EXPECT_EQ(join(tokenizer->tokenize("e\u0301")), std::string("e@@ \u0301"));
    std::string invalidUtf8 = "\xff\xfelow";
    EXPECT_EQ(tokenizer->detokenize(tokenizer->tokenize(invalidUtf8)), invalidUtf8);
}

TEST_F(Tokenizer, BpeTokenizerTest, Separator) {
    setTokenizerParameter("separator", "+");
    auto tokenizer = createTokenizer();
    auto tokens    = tokenizer->tokenize("wider tower");
    EXPECT_EQ(join(tokens), std::string("wide+ r tower"));
    EXPECT_EQ(tokenizer->detokenize(tokens), std::string("wider tower"));
}

// Vocabulary as written by `subword-nmt get-vocab`
TEST_F(Tokenizer, BpeTokenizerTest, LineVocabulary) {
    setTokenizerParameter("vocab-file", Test::dataFile("tokenizer/bpe.vocab"));
    auto tokenizer = createTokenizer();
    EXPECT_EQ(tokenizer->vocabularySize(), 32u);
    EXPECT_EQ(tokenizer->tokenStrToId("tower"), 1u);
    EXPECT_EQ(tokenizer->tokenStrToId("x@@"), AbstractTokenizer::invalidTokenId);
    EXPECT_EQ(tokenizer->tokenIdToStr(6u), std::string("newest"));

    std::string inVocabulary = "the newest lower tower is wider than straße";
    auto        ids          = tokenizer->encode(inVocabulary);
    EXPECT_EQ(join(ids), std::string("0 6 4 1 2 20 3 5 15 21 22 23 24 25"));
    EXPECT_EQ(tokenizer->decode(ids), inVocabulary);
}

// Vocabulary as written by create-py-vocab.py from rwth-i6/subword-nmt
TEST_F(Tokenizer, BpeTokenizerTest, DictVocabulary) {
    setTokenizerParameter("vocab-file", Test::dataFile("tokenizer/bpe.vocab.py"));
    setTokenizerParameter("unknown-token", "<unk>");
    auto tokenizer = createTokenizer();
    EXPECT_EQ(tokenizer->vocabularySize(), 34u);
    EXPECT_EQ(tokenizer->tokenStrToId("</s>"), 0u);
    EXPECT_EQ(tokenizer->tokenIdToStr(0u), std::string("<s>"));

    EXPECT_EQ(join(tokenizer->encode(text)), std::string("2 8 7 3 4 30 5 9 15 29 28 19 31 24 1 1"));
    EXPECT_EQ(tokenizer->decode({2u, 8u, 3u, 1u}), std::string("the newest tower <unk>"));
}

TEST_F(Tokenizer, BpeTokenizerTest, OutOfVocabulary) {
    setTokenizerParameter("vocab-file", Test::dataFile("tokenizer/bpe.vocab"));
    delayErrors();
    auto tokenizer = createTokenizer();
    EXPECT_EQ(join(tokenizer->encode("the xy")), std::string("0"));
    EXPECT_TRUE(tokenizer->hasFatalErrors());
}

TEST_F(Tokenizer, BpeTokenizerTest, InvalidTokenId) {
    setTokenizerParameter("vocab-file", Test::dataFile("tokenizer/bpe.vocab"));
    delayErrors();
    auto tokenizer = createTokenizer();
    EXPECT_EQ(tokenizer->tokenIdToStr(32u), std::string());
    EXPECT_TRUE(tokenizer->hasFatalErrors());
}

// Quotes and escape sequences as written by python's `repr`
TEST_F(Tokenizer, BpeTokenizerTest, DictVocabularyEscapes) {
    Test::Directory dir;
    setTokenizerParameter("vocab-file", writeFile(dir, "vocab.py", "{'<s>': 0, \"it's\": 1, 'a\\'b\\\\': 2, '\\u200b@@': 3, '\\xad': 4, 'ß': 5}\n"));
    auto tokenizer = createTokenizer();
    EXPECT_EQ(tokenizer->vocabularySize(), 6u);
    EXPECT_EQ(tokenizer->tokenStrToId("it's"), 1u);
    EXPECT_EQ(tokenizer->tokenStrToId("a'b\\"), 2u);
    EXPECT_EQ(tokenizer->tokenStrToId("\u200b@@"), 3u);
    EXPECT_EQ(tokenizer->tokenStrToId("\u00ad"), 4u);
    EXPECT_EQ(tokenizer->tokenStrToId("ß"), 5u);
}

// A line vocabulary is not mistaken for a python dict because its first token is "{"
TEST_F(Tokenizer, BpeTokenizerTest, LineVocabularyWithBraces) {
    Test::Directory dir;
    setTokenizerParameter("vocab-file", writeFile(dir, "vocab", "{ 10\n} 5\ntower 3\n"));
    auto tokenizer = createTokenizer();
    EXPECT_EQ(tokenizer->vocabularySize(), 3u);
    EXPECT_EQ(tokenizer->tokenStrToId("{"), 0u);
    EXPECT_EQ(tokenizer->tokenStrToId("tower"), 2u);
}

// Codes without version header are interpreted as version 0.1, i.e. the end-of-word marker is a separate symbol
TEST_F(Tokenizer, BpeTokenizerTest, CodesVersion01) {
    Test::Directory dir;
    setTokenizerParameter("codes-file", writeFile(dir, "codes", "l o\nlo w\ne r\ner </w>\nö n\nlo w\n"));
    auto tokenizer = createTokenizer();
    EXPECT_EQ(join(tokenizer->tokenize("lower low schön a lowlow")), std::string("low@@ er low s@@ c@@ h@@ ön a low@@ low"));
}

}  // namespace Tokenizer
