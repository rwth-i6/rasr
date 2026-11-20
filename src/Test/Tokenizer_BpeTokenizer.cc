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

// Expected values are taken from apply_bpe.py of rwth-i6/subword-nmt
class BpeTokenizerTest : public Test::TokenizerFixture {
public:
    virtual void setUp() {
        TokenizerFixture::setUp();
        setTokenizerParameter("type", "bpe");
        setTokenizerParameter("codes-file", Test::dataFile("tokenizer/bpe.codes"));
        setTokenizerParameter("vocab-file", Test::dataFile("tokenizer/bpe.vocab.py"));
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

// Whitespace characters of python's `str.split()` are word boundaries and are normalized to single blanks
TEST_F(Tokenizer, BpeTokenizerTest, Whitespace) {
    auto tokenizer = createTokenizer();
    auto tokens    = tokenizer->tokenize("the\ttower\nis  low\r\n");
    EXPECT_EQ(join(tokens), std::string("the tower is l@@ o@@ w"));
    EXPECT_EQ(tokenizer->detokenize(tokens), std::string("the tower is low"));
    EXPECT_EQ(join(tokenizer->tokenize("low\u00a0tower\u3000is")), std::string("l@@ o@@ w tower is"));
    // Zero-width spaces are no whitespace in python
    EXPECT_EQ(join(tokenizer->tokenize("low\u200btower")), std::string("l@@ o@@ w@@ \u200b@@ tower"));
}

TEST_F(Tokenizer, BpeTokenizerTest, Punctuation) {
    auto tokenizer = createTokenizer();
    auto tokens    = tokenizer->tokenize("lower. tower, (new)!");
    EXPECT_EQ(join(tokens), std::string("l@@ o@@ w@@ e@@ r@@ . t@@ o@@ w@@ e@@ r@@ , (@@ ne@@ w@@ )@@ !"));
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
    Test::Directory dir;
    setTokenizerParameter("separator", "+");
    setTokenizerParameter("vocab-file", writeFile(dir, "vocab.py", "{'wide+': 0, 'r': 1, 'tower': 2}"));
    auto tokenizer = createTokenizer();
    auto tokens    = tokenizer->tokenize("wider tower");
    EXPECT_EQ(join(tokens), std::string("wide+ r tower"));
    EXPECT_EQ(tokenizer->detokenize(tokens), std::string("wider tower"));
}

// Vocabulary as written by create-py-vocab.py of rwth-i6/subword-nmt
TEST_F(Tokenizer, BpeTokenizerTest, Vocabulary) {
    auto tokenizer = createTokenizer();
    EXPECT_EQ(tokenizer->vocabularySize(), 34u);
    EXPECT_EQ(tokenizer->tokenStrToId("tower"), 3u);
    EXPECT_EQ(tokenizer->tokenStrToId("x@@"), AbstractTokenizer::invalidTokenId);
    EXPECT_EQ(tokenizer->tokenIdToStr(3u), std::string("tower"));
    // "<s>" and "</s>" share an id, the first one is used for decoding
    EXPECT_EQ(tokenizer->tokenStrToId("</s>"), 0u);
    EXPECT_EQ(tokenizer->tokenIdToStr(0u), std::string("<s>"));

    std::string inVocabulary = "the newest lower tower is wider than straße";
    auto        ids          = tokenizer->encode(inVocabulary);
    EXPECT_EQ(join(ids), std::string("2 8 7 3 4 30 5 9 15 29 28 19 31 24"));
    EXPECT_EQ(tokenizer->decode(ids), inVocabulary);
}

TEST_F(Tokenizer, BpeTokenizerTest, UnknownToken) {
    setTokenizerParameter("unknown-token", "<unk>");
    auto tokenizer = createTokenizer();
    EXPECT_EQ(join(tokenizer->encode(text)), std::string("2 8 7 3 4 30 5 9 15 29 28 19 31 24 1 1"));
    EXPECT_EQ(tokenizer->decode({2u, 8u, 3u, 1u}), std::string("the newest tower <unk>"));
}

TEST_F(Tokenizer, BpeTokenizerTest, OutOfVocabulary) {
    delayErrors();
    auto tokenizer = createTokenizer();
    EXPECT_EQ(join(tokenizer->encode("the xy")), std::string("2"));
    EXPECT_TRUE(tokenizer->hasFatalErrors());
}

// As RETURNN does, subwords that are not in the vocabulary are split by reversing their merges, e.g. "owe" into "o"
// and "we" and then "we" into "w" and "e". Subwords that cannot be split further stay as they are.
TEST_F(Tokenizer, BpeTokenizerTest, VocabularySplit) {
    setTokenizerParameter("unknown-token", "<unk>");
    auto tokenizer = createTokenizer();
    EXPECT_EQ(join(tokenizer->tokenize("lower.")), std::string("l@@ o@@ w@@ e@@ r@@ ."));
    // The final subword "ower" is split via the merge of "owe" and "r</w>"
    EXPECT_EQ(join(tokenizer->tokenize("power")), std::string("p@@ o@@ w@@ e@@ r"));
    EXPECT_EQ(join(tokenizer->encode("power")), std::string("1 13 1 25 5"));
}

// In codes of version 0.1, the end-of-word marker is merged as separate symbol, e.g. "er" and "</w>"
TEST_F(Tokenizer, BpeTokenizerTest, VocabularySplitCodesVersion01) {
    Test::Directory dir;
    setTokenizerParameter("codes-file", writeFile(dir, "codes", "l o\nlo w\ne r\ner </w>\n"));
    setTokenizerParameter("vocab-file", writeFile(dir, "vocab.py", "{'l@@': 0, 'o@@': 1, 'w@@': 2, 'e@@': 3, 'r': 4}"));
    auto tokenizer = createTokenizer();
    EXPECT_EQ(join(tokenizer->tokenize("lower")), std::string("l@@ o@@ w@@ e@@ r"));
}

// Invalid ids are not decoded partially
TEST_F(Tokenizer, BpeTokenizerTest, DecodeInvalidTokenId) {
    delayErrors();
    auto tokenizer = createTokenizer();
    EXPECT_EQ(tokenizer->decode({2u, 34u}), std::string());
    EXPECT_TRUE(tokenizer->hasFatalErrors());
}

TEST_F(Tokenizer, BpeTokenizerTest, InvalidTokenId) {
    delayErrors();
    auto tokenizer = createTokenizer();
    EXPECT_EQ(tokenizer->tokenIdToStr(34u), std::string());
    EXPECT_TRUE(tokenizer->hasFatalErrors());
}

// All subwords except the last one of a word end with the separator, so only boundaries after tokens are marked
TEST_F(Tokenizer, BpeTokenizerTest, WordBoundaryMarks) {
    auto tokenizer = createTokenizer();
    EXPECT_EQ(formatWordBoundaryMarks(*tokenizer, tokenizer->encode("wider tower")), std::string("wide@@:-- r:-A tower:-A"));
}

TEST_F(Tokenizer, BpeTokenizerTest, WordBoundaryMarksInvalidTokenId) {
    delayErrors();
    auto tokenizer = createTokenizer();
    EXPECT_FALSE(tokenizer->wordBoundaryMarks(34u).after);
    EXPECT_TRUE(tokenizer->hasFatalErrors());
}

// Quotes and escape sequences as written by python's `repr`
TEST_F(Tokenizer, BpeTokenizerTest, VocabularyEscapes) {
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

// Codes without version header are interpreted as version 0.1, i.e. the end-of-word marker is a separate symbol
TEST_F(Tokenizer, BpeTokenizerTest, CodesVersion01) {
    Test::Directory dir;
    setTokenizerParameter("codes-file", writeFile(dir, "codes", "l o\nlo w\ne r\ner </w>\nö n\nlo w\n"));
    setTokenizerParameter("vocab-file", writeFile(dir, "vocab.py", "{'low@@': 0, 'er': 1, 'low': 2, 's@@': 3, 'c@@': 4, 'h@@': 5, 'ön': 6, 'a': 7}"));
    auto tokenizer = createTokenizer();
    EXPECT_EQ(join(tokenizer->tokenize("lower low schön a lowlow")), std::string("low@@ er low s@@ c@@ h@@ ön a low@@ low"));
}

}  // namespace Tokenizer
