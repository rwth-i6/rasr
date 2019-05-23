/** Copyright 2018 RWTH Aachen University. All rights reserved.
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
#include <Core/CompressedStream.hh>
#include <Core/Configuration.hh>
#include <Fsa/Application.hh>
#include <Fsa/Basic.hh>
#include <Translation/Reordering.hh>
#include <string>
#include <vector>
#include "ConditionalLexicon.hh"

class Aligner : public Core::Application {
private:
    static Core::ParameterInt         paramNBest_;
    static Core::ParameterFloatVector paramTransitionProbs_;
    static Core::ParameterFloat       paramFactorTransition_;
    static Core::ParameterChoice      paramLexiconType_;
    static Core::ParameterChoice      paramModel_;
    static Core::ParameterFloat       paramFactorReorder_;
    static Core::ParameterFloat       paramFactorLexicon_;
    static Core::ParameterFloat       paramFactorSourceLm_;
    static Core::ParameterInt         paramSegmentStart_;
    static Core::ParameterInt         paramNormalizePoint_;
    static Core::ParameterBool        paramNormalize_;
    static Core::ParameterInt         paramSegmentEnd_;
    static Core::ParameterString      paramPrune_;
    static Core::ParameterString      paramBeamPrune_;
    static Core::ParameterString      paramOutputXmlFilename_;
    static Core::ParameterString      paramOutputBiLangFilename_;
    static Core::ParameterString      paramOutputAachenFilename_;
    static Core::ParameterString      paramOutputLexiconFilename_;
    static Core::ParameterString      paramSourceLmFilename_;
    static Core::ParameterInt         paramOrder_;
    static Core::ParameterInt         paramExtractionOrder_;
    static Core::ParameterInt         paramIterations_;
    static Core::ParameterIntVector   paramIterationsOrder_;

    Fsa::Weight threshold_;
    Fsa::Weight beamThreshold_;
    float       totalCost_;

    u32 nbest_;

    unsigned                     order_, extractionOrder_;
    unsigned                     normalizePoint_;
    unsigned                     iterations_;
    bool                         writeXml_, writeBiLang_, writeAachen_, writeLexicon_;
    bool                         normalize_;
    bool                         useSourceLm_;
    std::string                  sourceLmFilename_;
    Fsa::ConstAutomatonRef       sourceLmAutomaton_;
    Core::CompressedOutputStream outputXml_, outputBiLang_, outputAachen_, outputLexicon_;

    std::vector<double> transitionProbs_;
    std::vector<int>    iterationsOrder_;

    Translation::Reordering* reordering_;
    bool                     reorderLinear_;
    double                   perplexity_, userTime_;
    Fsa::Weight              factorReorder_;
    Fsa::Weight              factorTransition_;
    Fsa::Weight              factorLexicon_;
    Fsa::Weight              factorSourceLm_;
    size_t                   sentence_, sentences_, oovs_, words_, arcs_, maxMemory_;
    u32                      segmentStart_, segmentEnd_;

    std::string getUsage() const;

public:
    Aligner();
    virtual ~Aligner();
    int         main(const std::vector<std::string>& arguments);
    std::string getConfiguration();
    void        setMemberVariablesFromParameters();

private:
    void processFile(const std::string& sourceFilename, const std::string& targetFilename);

    Fsa::ConstAutomatonRef createAlignAutomaton(std::string&                            sourceSentence,
                                                std::string&                            targetSentence,
                                                Translation::ConstConditionalLexiconRef conditionalLexicon);

    Fsa::ConstAutomatonRef createAlignment(std::string&                            sourceSentence,
                                           std::string&                            targetSentence,
                                           Translation::ConstConditionalLexiconRef conditionalLexicon,
                                           Translation::Reordering*                reordering);

    void extract(Translation::ConditionalLexiconRef lexicon, Fsa::ConstAutomatonRef f, int extractionOrder);
};
