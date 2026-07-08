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

#include <Bliss/CorpusDescription.hh>
#include <Bliss/Orthography.hh>
#include <Test/File.hh>
#include <Test/UnitTest.hh>

namespace Bliss {

class OrthographyParserVisitor : public CorpusVisitor {
public:
    virtual void visitSpeechSegment(SpeechSegment* segment) {
        orthographies_.push_back(segment->orthography());
        leftContextOrthographies_.push_back(segment->leftContextOrthography());
        rightContextOrthographies_.push_back(segment->rightContextOrthography());
    }

    std::vector<Orthography> const& orthographies() const {
        return orthographies_;
    }

    std::vector<Orthography> const& leftContextOrthographies() const {
        return leftContextOrthographies_;
    }

    std::vector<Orthography> const& rightContextOrthographies() const {
        return rightContextOrthographies_;
    }

private:
    std::vector<Orthography> orthographies_;
    std::vector<Orthography> leftContextOrthographies_;
    std::vector<Orthography> rightContextOrthographies_;
};

class OrthographyParserTest : public Test::ConfigurableFixture {
public:
    virtual void setUp() {
        tmpDir_     = std::make_unique<Test::Directory>();
        corpusFile_ = Test::File(*tmpDir_, "test.corpus").path();
        setParameter("*.channel", "nil");
        setParameter("*.error.channel", "stderr");
        setParameter("*.corpus.file", corpusFile_);
    }

protected:
    Orthography parseOrth(std::string const& orthXml) {
        OrthographyParserVisitor visitor;
        parseCorpus("<segment>" + orthXml + "</segment>", visitor);
        EXPECT_EQ(visitor.orthographies().size(), size_t(1));
        return visitor.orthographies().front();
    }

    void parseCorpus(std::string const& segmentContent, OrthographyParserVisitor& visitor) {
        std::ofstream os(corpusFile_.c_str());
        os << "<corpus name=\"test\"><recording name=\"rec\" audio=\"none\">"
           << segmentContent
           << "</recording></corpus>";
        os.close();

        CorpusDescription description(select("corpus"));
        description.accept(&visitor);
    }

    std::unique_ptr<Test::Directory> tmpDir_;
    std::string                      corpusFile_;
};

TEST(Bliss, Orthography, SingleTextSpan) {
    Orthography orth = Orthography::fromNormalized("hello world ");

    EXPECT_EQ(orth.str(), std::string("hello world "));
    EXPECT_FALSE(orth.empty());
    EXPECT_EQ(orth.spans().size(), size_t(1));
    EXPECT_EQ(orth.spans().front().type(), Orthography::Span::Type::text);
    EXPECT_EQ(orth.spans().front().text(), std::string("hello world "));
}

TEST(Bliss, Orthography, MultipleTextSpans) {
    Orthography orth;
    orth.appendText("hello ");
    orth.appendText("world ");

    EXPECT_EQ(orth.str(), std::string("hello world "));
    EXPECT_EQ(orth.spans().size(), size_t(2));
}

TEST(Bliss, Orthography, AlternativeSpanUsesFirstAlternative) {
    std::vector<Orthography> alternatives;
    alternatives.push_back(Orthography::fromNormalized("first path "));
    alternatives.push_back(Orthography::fromNormalized("second path "));

    Orthography orth;
    orth.appendText("prefix ");
    orth.appendAlternative(alternatives);
    orth.appendText("suffix ");

    // Orthography::str() keeps the historical single-string behavior by
    // rendering each alternatives span through its first alternative.
    EXPECT_EQ(orth.str(), std::string("prefix first path suffix "));
    EXPECT_EQ(orth.spans().size(), size_t(3));
    EXPECT_EQ(orth.spans()[1].type(), Orthography::Span::Type::alternatives);
    EXPECT_EQ(orth.spans()[1].alternatives().size(), size_t(2));
}

TEST(Bliss, Orthography, NestedAlternatives) {
    std::vector<Orthography> innerAlternatives;
    innerAlternatives.push_back(Orthography::fromNormalized("inner first "));
    innerAlternatives.push_back(Orthography::fromNormalized("inner second "));

    Orthography nested;
    nested.appendText("nested ");
    nested.appendAlternative(innerAlternatives);

    std::vector<Orthography> outerAlternatives;
    outerAlternatives.push_back(nested);
    outerAlternatives.push_back(Orthography::fromNormalized("outer second "));

    Orthography orth;
    orth.appendAlternative(outerAlternatives);

    EXPECT_EQ(orth.str(), std::string("nested inner first "));
}

TEST(Bliss, Orthography, ClearAndEmpty) {
    Orthography orth = Orthography::fromNormalized("text ");
    EXPECT_FALSE(orth.empty());

    orth.clear();
    EXPECT_TRUE(orth.empty());
    EXPECT_EQ(orth.str(), std::string(""));
    EXPECT_EQ(orth.spans().size(), size_t(0));
}

TEST_F(Bliss, OrthographyParserTest, PlainOrthCompatibility) {
    Orthography orth = parseOrth("<orth>hello <noise>world</noise></orth>");

    EXPECT_EQ(orth.str(), std::string("hello world "));
    EXPECT_EQ(orth.spans().size(), size_t(1));
    EXPECT_EQ(orth.spans().front().type(), Orthography::Span::Type::text);
    EXPECT_EQ(orth.spans().front().text(), std::string("hello world "));
}

TEST_F(Bliss, OrthographyParserTest, Alternatives) {
    Orthography orth = parseOrth(
            "<orth>prefix <alternatives>"
            "<orth>first choice</orth>"
            "<orth>second choice</orth>"
            "</alternatives> suffix</orth>");

    EXPECT_EQ(orth.str(), std::string("prefix first choice suffix "));
    EXPECT_EQ(orth.spans().size(), size_t(3));
    EXPECT_EQ(orth.spans()[0].text(), std::string("prefix "));
    EXPECT_EQ(orth.spans()[1].type(), Orthography::Span::Type::alternatives);
    EXPECT_EQ(orth.spans()[1].alternatives().size(), size_t(2));
    EXPECT_EQ(orth.spans()[1].alternatives()[0].str(), std::string("first choice "));
    EXPECT_EQ(orth.spans()[1].alternatives()[1].str(), std::string("second choice "));
    EXPECT_EQ(orth.spans()[2].text(), std::string("suffix "));
}

TEST_F(Bliss, OrthographyParserTest, EmptyAlternative) {
    Orthography orth = parseOrth(
            "<orth><alternatives>"
            "<orth>optional context</orth>"
            "<orth/>"
            "</alternatives></orth>");

    EXPECT_EQ(orth.str(), std::string("optional context "));
    EXPECT_EQ(orth.spans().size(), size_t(1));
    EXPECT_EQ(orth.spans()[0].alternatives().size(), size_t(2));
    EXPECT_EQ(orth.spans()[0].alternatives()[0].str(), std::string("optional context "));
    EXPECT_EQ(orth.spans()[0].alternatives()[1].str(), std::string(""));
}

TEST_F(Bliss, OrthographyParserTest, NestedAlternatives) {
    Orthography orth = parseOrth(
            "<orth><alternatives>"
            "<orth>outer <alternatives><orth>inner one</orth><orth>inner two</orth></alternatives></orth>"
            "<orth>fallback</orth>"
            "</alternatives></orth>");

    EXPECT_EQ(orth.str(), std::string("outer inner one "));
    EXPECT_EQ(orth.spans().size(), size_t(1));
    Orthography const& firstAlternative = orth.spans()[0].alternatives()[0];
    // Alternatives hold complete Orthography objects, so nested alternatives
    // are represented structurally instead of flattened into text.
    EXPECT_EQ(firstAlternative.spans().size(), size_t(2));
    EXPECT_EQ(firstAlternative.spans()[1].type(), Orthography::Span::Type::alternatives);
    EXPECT_EQ(firstAlternative.spans()[1].alternatives()[1].str(), std::string("inner two "));
}

TEST_F(Bliss, OrthographyParserTest, Optional) {
    // <optional>text</optional> is parser shorthand for
    // <alternatives><orth>text</orth><orth/></alternatives>.
    Orthography orth = parseOrth("<orth>prefix <optional>maybe</optional> suffix</orth>");

    EXPECT_EQ(orth.str(), std::string("prefix maybe suffix "));
    EXPECT_EQ(orth.spans().size(), size_t(3));
    EXPECT_EQ(orth.spans()[0].text(), std::string("prefix "));
    EXPECT_EQ(orth.spans()[1].type(), Orthography::Span::Type::alternatives);
    EXPECT_EQ(orth.spans()[1].alternatives().size(), size_t(2));
    EXPECT_EQ(orth.spans()[1].alternatives()[0].str(), std::string("maybe "));
    EXPECT_EQ(orth.spans()[1].alternatives()[1].str(), std::string(""));
    EXPECT_EQ(orth.spans()[2].text(), std::string("suffix "));
}

TEST_F(Bliss, OrthographyParserTest, NestedOptional) {
    // The first optional alternative contains a full Orthography, therefore
    // nested <optional> elements become nested alternatives spans.
    Orthography orth = parseOrth("<orth><optional>outer <optional>inner</optional></optional></orth>");

    EXPECT_EQ(orth.str(), std::string("outer inner "));
    EXPECT_EQ(orth.spans().size(), size_t(1));
    EXPECT_EQ(orth.spans()[0].type(), Orthography::Span::Type::alternatives);
    EXPECT_EQ(orth.spans()[0].alternatives().size(), size_t(2));
    Orthography const& firstAlternative = orth.spans()[0].alternatives()[0];
    EXPECT_EQ(firstAlternative.spans().size(), size_t(2));
    EXPECT_EQ(firstAlternative.spans()[0].text(), std::string("outer "));
    EXPECT_EQ(firstAlternative.spans()[1].type(), Orthography::Span::Type::alternatives);
    EXPECT_EQ(firstAlternative.spans()[1].alternatives().size(), size_t(2));
    EXPECT_EQ(firstAlternative.spans()[1].alternatives()[0].str(), std::string("inner "));
    EXPECT_EQ(firstAlternative.spans()[1].alternatives()[1].str(), std::string(""));
}

TEST_F(Bliss, OrthographyParserTest, ContextOrthographiesRemainPlain) {
    OrthographyParserVisitor visitor;
    // Context orthographies are still parsed by the legacy plain-text path:
    // child element text is kept, but no structured alternatives are created.
    parseCorpus(
            "<segment>"
            "<orth>main</orth>"
            "<left-context-orth>left <alternatives><orth>ignored tag</orth></alternatives> <optional>plain optional</optional></left-context-orth>"
            "<right-context-orth>right</right-context-orth>"
            "</segment>",
            visitor);

    EXPECT_EQ(visitor.orthographies().size(), size_t(1));
    EXPECT_EQ(visitor.orthographies().front().str(), std::string("main "));
    EXPECT_EQ(visitor.leftContextOrthographies().front().str(), std::string("left ignored tag plain optional "));
    EXPECT_EQ(visitor.leftContextOrthographies().front().spans().size(), size_t(1));
    EXPECT_EQ(visitor.rightContextOrthographies().front().str(), std::string("right "));
}

}  // namespace Bliss
