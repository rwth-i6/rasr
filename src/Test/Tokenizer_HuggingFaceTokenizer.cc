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
#include <pybind11/pybind11.h>

#include <Core/Application.hh>
#include <Python/Init.hh>

#include <Test/TokenizerFixture.hh>

namespace py = pybind11;

namespace Tokenizer {

// Expected values are taken from the tokenizers python package
class HuggingFaceTokenizerTest : public Test::TokenizerFixture {
public:
    virtual void setUp() {
        TokenizerFixture::setUp();
        setTokenizerParameter("type", "huggingface");
        setTokenizerParameter("tokenizer-file", Test::dataFile("tokenizer/tokenizer.json"));
    }

protected:
    // A tokenizer.json with a word-level model and the special token "[S]"
    static std::string minimalTokenizer(std::string const& decoder) {
        return R"({"version": "1.0", "truncation": null, "padding": null, "normalizer": null, "pre_tokenizer": null, "post_processor": null,
                   "added_tokens": [{"id": 4, "content": "[S]", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}],
                   "model": {"type": "WordLevel", "vocab": {"▁a": 0, "b": 1, "##c": 2, "d</w>": 3, "[S]": 4}, "unk_token": "b"},
                   "decoder": )" +
               decoder + "}";
    }

    // The tests are skipped if the python package is not installed
    static bool tokenizersAvailable() {
        static bool available = [] {
            // Kept alive until exit since python cannot be re-initialized reliably (see HuggingFaceTokenizer.cc)
            static Python::Initializer initializer;
            Core::Application::us()->atexit([] { initializer.uninit(); });
            initializer.init();
            py::gil_scoped_acquire gil;
            try {
                py::module_::import("tokenizers");
                return true;
            }
            catch (py::error_already_set const&) {
                std::cerr << "Skipping test: python package 'tokenizers' is not available" << std::endl;
                return false;
            }
        }();
        return available;
    }
};

TEST_F(Tokenizer, HuggingFaceTokenizerTest, Tokenize) {
    if (not tokenizersAvailable()) {
        return;
    }
    auto tokenizer = createTokenizer();
    auto tokens    = tokenizer->tokenize(text);
    EXPECT_EQ(join(tokens), std::string("the Ġnewest Ġlower Ġtower Ġis Ġwide r Ġthan Ġ st raÃ Ł e Ġ x y"));
    EXPECT_EQ(tokenizer->detokenize(tokens), text);
}

TEST_F(Tokenizer, HuggingFaceTokenizerTest, Vocabulary) {
    if (not tokenizersAvailable()) {
        return;
    }
    auto tokenizer = createTokenizer();
    EXPECT_EQ(tokenizer->vocabularySize(), 300u);
    EXPECT_EQ(tokenizer->tokenStrToId("<s>"), 0u);
    EXPECT_EQ(tokenizer->tokenStrToId("Ġtower"), 266u);
    EXPECT_EQ(tokenizer->tokenStrToId("xy"), AbstractTokenizer::invalidTokenId);
    EXPECT_EQ(tokenizer->tokenIdToStr(266u), std::string("Ġtower"));
}

TEST_F(Tokenizer, HuggingFaceTokenizerTest, EncodeDecode) {
    if (not tokenizersAvailable()) {
        return;
    }
    auto tokenizer = createTokenizer();
    auto ids       = tokenizer->encode(text);
    EXPECT_EQ(join(ids), std::string("277 283 281 266 271 279 83 280 222 262 297 255 70 222 89 90"));
    EXPECT_EQ(tokenizer->decode(ids), text);
}

// The post-processor of the test tokenizer prepends "<s>"
TEST_F(Tokenizer, HuggingFaceTokenizerTest, SpecialTokens) {
    if (not tokenizersAvailable()) {
        return;
    }
    setTokenizerParameter("add-special-tokens-on-encode", "true");
    setTokenizerParameter("remove-special-tokens-on-decode", "false");
    auto tokenizer = createTokenizer();
    auto ids       = tokenizer->encode("the tower");
    EXPECT_EQ(join(ids), std::string("0 277 266"));
    EXPECT_EQ(tokenizer->decode(ids), std::string("<s>the tower"));

    setTokenizerParameter("remove-special-tokens-on-decode", "true");
    EXPECT_EQ(createTokenizer()->decode(ids), std::string("the tower"));
}

TEST_F(Tokenizer, HuggingFaceTokenizerTest, EmptyInput) {
    if (not tokenizersAvailable()) {
        return;
    }
    auto tokenizer = createTokenizer();
    EXPECT_TRUE(tokenizer->tokenize("").empty());
    EXPECT_TRUE(tokenizer->encode("").empty());
    EXPECT_EQ(tokenizer->detokenize({}), std::string());
    EXPECT_EQ(tokenizer->decode({}), std::string());
}

// Byte-level BPE keeps all whitespace, e.g. "Ġ" for blanks and "Ċ" for newlines
TEST_F(Tokenizer, HuggingFaceTokenizerTest, Whitespace) {
    if (not tokenizersAvailable()) {
        return;
    }
    auto        tokenizer = createTokenizer();
    std::string text      = "the\ttower\nis  low";
    auto        tokens    = tokenizer->tokenize(text);
    EXPECT_EQ(join(tokens), std::string("the ĉ t ower Ċ is Ġ Ġl o w"));
    EXPECT_EQ(tokenizer->detokenize(tokens), text);
    EXPECT_EQ(tokenizer->decode(tokenizer->encode("   ")), std::string("   "));
}

TEST_F(Tokenizer, HuggingFaceTokenizerTest, Punctuation) {
    if (not tokenizersAvailable()) {
        return;
    }
    auto        tokenizer = createTokenizer();
    std::string text      = " lower. tower, (new)! ";
    auto        tokens    = tokenizer->tokenize(text);
    EXPECT_EQ(join(tokens), std::string("Ġlower . Ġtower , Ġ ( ne w ) ! Ġ"));
    EXPECT_EQ(tokenizer->detokenize(tokens), text);
}

// Characters outside of the training data are split into bytes
TEST_F(Tokenizer, HuggingFaceTokenizerTest, RareCharacters) {
    if (not tokenizersAvailable()) {
        return;
    }
    auto tokenizer = createTokenizer();
    EXPECT_EQ(join(tokenizer->tokenize("🙂 tower")), std::string("ð Ł Ļ Ĥ Ġtower"));
    EXPECT_EQ(tokenizer->decode(tokenizer->encode("🙂 e\u0301")), std::string("🙂 e\u0301"));
}

// Python cannot represent invalid UTF-8 as str, which results in an error instead of an exception
TEST_F(Tokenizer, HuggingFaceTokenizerTest, InvalidUtf8) {
    if (not tokenizersAvailable()) {
        return;
    }
    delayErrors();
    auto tokenizer = createTokenizer();
    EXPECT_TRUE(tokenizer->encode("\xff\xfelow").empty());
    EXPECT_TRUE(tokenizer->hasFatalErrors());
}

TEST_F(Tokenizer, HuggingFaceTokenizerTest, InvalidTokenId) {
    if (not tokenizersAvailable()) {
        return;
    }
    delayErrors();
    auto tokenizer = createTokenizer();
    EXPECT_EQ(tokenizer->tokenIdToStr(300u), std::string());
    EXPECT_EQ(tokenizer->decode({277u, 300u}), std::string());
    EXPECT_TRUE(tokenizer->hasFatalErrors());
}

// Byte-level BPE marks word starts with "Ġ", whitespace such as newlines and special tokens are words of their own
TEST_F(Tokenizer, HuggingFaceTokenizerTest, WordBoundaryMarks) {
    if (not tokenizersAvailable()) {
        return;
    }
    setTokenizerParameter("add-special-tokens-on-encode", "true");
    auto tokenizer = createTokenizer();
    EXPECT_EQ(formatWordBoundaryMarks(*tokenizer, tokenizer->encode("the tower,\nlow")), std::string("<s>:BA the:-- Ġtower:B- ,:-- Ċ:BA lo:-- w:--"));
}

// The marking is determined from the decoder, here tested with minimal tokenizers
TEST_F(Tokenizer, HuggingFaceTokenizerTest, WordBoundaryDecoders) {
    if (not tokenizersAvailable()) {
        return;
    }
    std::vector<std::pair<std::string, std::string>> decoders = {
            {R"({"type": "Metaspace", "replacement": "▁", "prepend_scheme": "always", "split": true})", "▁a:B- b:-- ##c:-- d</w>:-- [S]:BA"},
            {R"({"type": "Sequence", "decoders": [{"type": "Replace", "pattern": {"String": "▁"}, "content": " "}, {"type": "Fuse"}]})", "▁a:B- b:-- ##c:-- d</w>:-- [S]:BA"},
            {R"({"type": "WordPiece", "prefix": "##", "cleanup": true})", "▁a:B- b:B- ##c:-- d</w>:B- [S]:BA"},
            {R"({"type": "BPEDecoder", "suffix": "</w>"})", "▁a:-- b:-- ##c:-- d</w>:-A [S]:BA"},
    };
    Test::Directory dir;
    for (auto const& [decoder, expected] : decoders) {
        setTokenizerParameter("tokenizer-file", writeFile(dir, "tokenizer.json", minimalTokenizer(decoder)));
        EXPECT_EQ(formatWordBoundaryMarks(*createTokenizer(), {0u, 1u, 2u, 3u, 4u}), expected);
    }
}

TEST_F(Tokenizer, HuggingFaceTokenizerTest, WordBoundaryMarksUnknownDecoder) {
    if (not tokenizersAvailable()) {
        return;
    }
    Test::Directory dir;
    setTokenizerParameter("tokenizer-file", writeFile(dir, "tokenizer.json", minimalTokenizer("null")));
    delayErrors();
    auto tokenizer = createTokenizer();
    EXPECT_TRUE(tokenizer->wordBoundaryMarks(4u).before);  // Special tokens do not depend on the decoder
    tokenizer->wordBoundaryMarks(0u);
    EXPECT_TRUE(tokenizer->hasFatalErrors());
}

}  // namespace Tokenizer
