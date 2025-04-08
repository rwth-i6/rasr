/** Copyright 2020 RWTH Aachen University. All rights reserved.
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
// $Id$

#ifndef _BLISS_LEXICONPARSER_HH
#define _BLISS_LEXICONPARSER_HH

#include <Core/Hash.hh>
#include <Core/XmlBuilder.hh>
#include "Lexicon.hh"

namespace Bliss {

class PhonemeInventoryElement : public Core::XmlBuilderElement<
                                        PhonemeInventory,
                                        Core::XmlRegularElement,
                                        Core::CreateUsingNew> {
    typedef Core::XmlBuilderElement<
            PhonemeInventory,
            Core::XmlRegularElement,
            Core::CreateUsingNew>
                                    Precursor;
    typedef PhonemeInventoryElement Self;

private:
    Phoneme* phoneme_;

    void startPhonemedef(const Core::XmlAttributes atts);
    void endPhonemedef();
    void phonemedefSymbol(const std::string&);
    void phonemedefVariation(const std::string&);

public:
    PhonemeInventoryElement(Core::XmlContext* _context, Handler _handler = 0);
    virtual void characters(const char*, int);
};

struct WeightedPhonemeString;
class PronunciationElement;
class LexiconElement;
class LexiconParser;
class TextLexiconParser;
class XmlLexiconParser;

class LexiconElement : public Core::XmlBuilderElement<
                               Lexicon,
                               Core::XmlRegularElement,
                               Core::CreateByContext> {
    friend class LexiconParser;
    friend class XmlLexiconParser;
    typedef Core::XmlBuilderElement<
            Lexicon,
            Core::XmlRegularElement,
            Core::CreateByContext>
                           Precursor;
    typedef LexiconElement Self;

private:
    static const Core::ParameterBool paramNormalizePronunciation;

    Lexicon*                 lexicon_;
    Core::StringHashSet      whitelist_;
    Lemma*                   lemma_;
    std::string              lemmaName_;
    std::string              specialLemmaName_;
    std::vector<std::string> orths_;
    std::vector<std::string> tokSeq_;

    void addPhonemeInventory(std::unique_ptr<PhonemeInventory>&);
    void startLemma(const Core::XmlAttributes atts);
    void addOrth(const std::string&);
    void addPhon(const WeightedPhonemeString&);
    void startTokSeq(const Core::XmlAttributes atts);
    void tok(const std::string&);
    void endTokSeq();
    void startSynt(const Core::XmlAttributes atts);
    void syntTok(const std::string&);
    void endSynt();
    void startEval(const Core::XmlAttributes atts);
    void evalTok(const std::string&);
    void endEval();
    void endLemma();
    bool isNormalizePronunciation_;

public:
    LexiconElement(Core::XmlContext*, CreationHandler, const Core::Configuration& c);
    virtual void characters(const char*, int) {};
};

/*
 * Base lexicon parser class
 */
class LexiconParser {
public:
    virtual ~LexiconParser() {}
    virtual bool     parseFile(const std::string& filename) = 0;
    virtual Lexicon* lexicon() const                        = 0;
};

/**
 * Parser for Bliss lexicon files.
 * This class implements parsing of the lexicon XML format
 * described in <a href="../../doc/Lexicon.pdf">Lexicon File
 * Format Reference</a>.  It is normally not used directly but
 * through Lexicon.
 */
class XmlLexiconParser : public virtual LexiconParser, public Core::XmlSchemaParser {
    typedef XmlLexiconParser Self;

private:
    Lexicon* lexicon_;
    Lexicon* pseudoCreateLexicon(Core::XmlAttributes) {
        return lexicon_;
    }
    void loadWhitelist(const Core::Configuration&, Core::StringHashSet&);

public:
    XmlLexiconParser(const Core::Configuration& c, Lexicon*);
    bool     parseFile(const std::string& filename) override;
    Lexicon* lexicon() const override {
        return lexicon_;
    }
};

struct XmlLexiconFormat : public Core::FormatSet::Format<Lexicon*> {
    bool read(const std::string& filename, Lexicon*& lexicon) const override {
        XmlLexiconParser parser(Core::Application::us()->getConfiguration(), lexicon);
        return parser.parseFile(filename);
    }

    bool write(const std::string& filename, Bliss::Lexicon* const& lexicon) const override {
        return false;
    }
};

/**
 * Parser for text lexicon files containing the vocab, so only the labels
 * This is meant for "lexicon-free" search
 * The .txt-file should contain one label per line
 */
class VocabTextLexiconParser : public LexiconParser {
private:
    Lexicon*          lexicon_;
    PhonemeInventory* phonemeInventory_;
    void              createPhoneme(const std::string& line);
    void              createLemmata();

public:
    VocabTextLexiconParser(Lexicon*);
    bool     parseFile(const std::string& filename) override;
    Lexicon* lexicon() const override {
        return lexicon_;
    }
};

struct VocabTextLexiconFormat : public Core::FormatSet::Format<Lexicon*> {
    bool read(const std::string& filename, Lexicon*& lexicon) const override {
        VocabTextLexiconParser parser(lexicon);
        return parser.parseFile(filename);
    }

    bool write(const std::string& filename, Bliss::Lexicon* const& lexicon) const override {
        return false;
    }
};

}  // namespace Bliss

#endif  // _BLISS_LEXICONPARSER_HH
