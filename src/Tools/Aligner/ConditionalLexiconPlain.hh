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
#ifndef CONDITIONAL_LEXICON_PLAIN_HH
#define CONDITIONAL_LEXICON_PLAIN_HH

#include <Core/Component.hh>
#include <Core/CompressedStream.hh>
#include <Core/ReferenceCounting.hh>
#include <Fsa/Automaton.hh>
#include <Translation/Common.hh>
#include <Translation/PrefixTree.hh>
#include <vector>
#include "ConditionalLexicon.hh"

#include <iostream>

namespace Translation {

class ConditionalLexiconPlain : public ConditionalLexicon {
public:
    ConditionalLexiconPlain(const Core::Configuration& config)
            : ConditionalLexicon(config),
              lexica_(),
              lexiconFilename_(paramFilename_(config)),
              floor_(paramFloor_(config)) {
        if (lexiconFilename_ != "") {
            std::cerr << "lexicon filename: " << lexiconFilename_ << std::endl;
            std::cerr << "reading ..." << std::endl;
            Core::CompressedInputStream lexiconInputStream_(lexiconFilename_);
            this->read(lexiconInputStream_);
        }
    }

    ConditionalLexiconPlain(const Core::Configuration& config, Fsa::ConstAlphabetRef alphabet)
            : ConditionalLexicon(config, alphabet),
              lexiconFilename_(paramFilename_(config)),
              floor_(paramFloor_(config)) {
        if (lexiconFilename_ != "") {
            std::cerr << "lexicon filename: " << lexiconFilename_ << std::endl;
            std::cerr << "reading ..." << std::endl;
            Core::CompressedInputStream lexiconInputStream_(lexiconFilename_);
            this->read(lexiconInputStream_);
        }
    }

    //! this is deprecated and should just exist as long as i am migrating to the new lexicon
    virtual Translation::Cost getProb(const size_t index, const std::vector<std::string>& key) const {
        std::vector<Fsa::LabelId> mappedKey;
        for (size_t i = 0; i < key.size(); ++i) {
            mappedKey.push_back(tokens_->index(key[i]));
        }
        return getProb(index, mappedKey);
    }

    virtual Translation::Cost getCost(const size_t index, const std::vector<Fsa::LabelId>& key) const {
        bool                                      exists      = true;
        const Lexicon::Node*                      currentNode = lexica_[index]->rootNode();
        std::vector<Fsa::LabelId>::const_iterator i           = key.begin();
        while (exists && i != key.end()) {
            currentNode = currentNode->follow(*i);
            if (currentNode == 0) {
                exists = false;
            }
            ++i;
        }

        Translation::Cost result;
        if (exists) {
            result = currentNode->getData();
        }
        else {
            result = floor_;
        }
        return result;
    }

    virtual Translation::Cost getReverseCost(const size_t index, const std::vector<Fsa::LabelId>& key) const {
        bool exists = true;

        std::vector<Fsa::LabelId> reverseKey;

        for (int j = key.size() - 1; j > -1; j--) {
            if (j % 2 == 0) {
                reverseKey.push_back(key[j]);
                reverseKey.push_back(key[j + 1]);
            }
        }

        const Lexicon::Node*                      currentNode = lexica_[index]->rootNode();
        std::vector<Fsa::LabelId>::const_iterator i           = reverseKey.begin();
        while (exists && i != reverseKey.end()) {
            currentNode = currentNode->follow(*i);
            if (currentNode == 0) {
                exists = false;
            }
            ++i;
        }

        Translation::Cost result;
        if (exists) {
            result = currentNode->getData();
        }
        else {
            result = floor_;
        }
        return result;
    }

    //! get probability of a lexicon entry or floor if it does not exist
    virtual Translation::Cost getProb(const size_t index, const std::vector<Fsa::LabelId>& key) const {
        return ::exp(-(this->getCost(index, key)));
    }

    virtual void addValue(const size_t index, const std::vector<Fsa::LabelId>& key, Translation::Cost value) {
        /*
         * traverse tree to store key and create arcs that do not exist so far.
         * existence has to be memorized in order to decide whether to store or
         * to increase the current key.
         */

        if (lexica_.size() < index + 1) {
            //initialize new lexica
            size_t oldSize = lexica_.size();
            lexica_.resize(index + 1);
            for (size_t i = oldSize; i <= index; ++i) {
                lexica_[i] = LexiconRef(new Lexicon);
            }
        }

        Lexicon::Node* currentNode = lexica_[index]->rootNode();
        currentNode->setData(currentNode->getData() + value);
        std::vector<Fsa::LabelId>::const_iterator i = key.begin();
        while (i != key.end()) {
            Lexicon::Node* newNode = currentNode->follow(*i);
            if (newNode == 0) {
                newNode = currentNode->followOrExpand(*i);
                newNode->setData(value);
            }

            else {
                newNode->setData(newNode->getData() + value);
            }

            currentNode = newNode;
            ++i;
        }
    }

    //! add value to existing count/prob or create new if it does not exist
    virtual void addValue(const size_t index, const std::vector<std::string>& key, Translation::Cost value) {
        std::vector<Fsa::LabelId> labelIdKey;
        for (std::vector<std::string>::const_iterator i = key.begin(); i != key.end(); ++i) {
            labelIdKey.push_back(tokens_->addSymbol(*i));
        }
        this->addValue(index, labelIdKey, value);
    };

    //! set value of the given entry (overwrite if it exists, create if it doesnt)
    virtual void setValue(const size_t index, const std::vector<Fsa::LabelId>& key, Translation::Cost value) {
        lexica_[index]->store(key, value);
    }

    //! set value of the given entry (overwrite if it exists, create if it doesnt)
    virtual void setValue(const size_t index, const std::vector<std::string>& key, Translation::Cost value){};

    //! write lexicon to stream
    virtual void write(std::ostream& out) {
        for (uint index = 0; index < lexica_.size(); ++index) {
            for (Lexicon::iterator it = lexica_[index]->begin(); it != lexica_[index]->end(); ++it) {
                std::vector<Fsa::LabelId> path = it.getIndexPath();
                if (it->isLeaf()) {
                    out << index << " "
                        << "prob: " << exp(-(it->getData())) << " cost: " << it->getData() << " ";
                    for (uint i = 0; i < path.size(); i++) {
                        out << tokens_->symbol(path[i]) << " ";
                    }
                    out << std::endl;
                }
            }
        }
    }

    //! normalize
    virtual void normalize(int normalizePoint) {
        int steps = normalizePoint;

        for (uint index = 0; index < lexica_.size(); ++index) {
            for (Lexicon::iterator it = lexica_[index]->begin(); it != lexica_[index]->end(); ++it) {
                std::vector<Fsa::LabelId> path = it.getIndexPath();
                if (it->isLeaf()) {
                    Lexicon::Node* currentNode = *it;
                    while (steps != 0 && currentNode != lexica_[index]->rootNode()) {
                        steps--;
                        currentNode = currentNode->up();
                    }

                    it->setData(-::log(it->getData() / currentNode->getData()));
                    steps = normalizePoint;
                }
            }
        }
    }

private:
    typedef Translation::SimplePrefixTree<Fsa::LabelId, Translation::Cost> Lexicon;
    typedef Core::Ref<Lexicon>                                             LexiconRef;

    //! holds lexica for the different types of transitions (if neccessary)
    std::vector<LexiconRef> lexica_;

    //! parameter giving the filename to read from
    static Core::ParameterString paramFilename_;

    //! filename as a string. read from the corresponding parameter
    const std::string lexiconFilename_;

    //! floor value for this lexicon
    static Core::ParameterFloat paramFloor_;

    //! floor probability for values that are not in the lexicon
    Translation::Cost floor_;

public:
    //! read a lexicon from a file stream
    virtual void read(std::istream&);
    //! read a lexicon from a file stream
    virtual void read();
};
}  // namespace Translation
#endif
