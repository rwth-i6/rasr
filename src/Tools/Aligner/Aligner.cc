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
#include "Aligner.hh"

#include <cmath>

#include <Fsa/Alphabet.hh>
#include <Fsa/Arithmetic.hh>
#include <Fsa/Best.hh>
#include <Fsa/Compose.hh>
#include <Fsa/Output.hh>
#include <Fsa/Prune.hh>
#include <Fsa/Static.hh>
#include <Fsa/Storage.hh>
#include <Fsa/Input.hh>

#include "Common.hh"
#include <Translation/Common.hh>

#include "ConditionalLexiconPlain.hh"
#include "ConditionalLexiconSri.hh"

#include "SimpleAlignAutomaton.hh"

#include <Fsa/Output.hh>
using namespace Fsa;
using __gnu_cxx::hash_map;
using __gnu_cxx::hash;

APPLICATION(Aligner)

std::string Aligner::getUsage() const {
        std::string usage = "\n"
                "aligner [OPTION(S)] <lexicon> <source> <target>\n"
                "\n"
                "aligner options:\n"
                "   --startSentence=<n>         set start sentence\n"
                "   --endSentence=<n>           set end sentence\n"
                "   --transitionProbs=<p>,<p>,<p>  transition probabilities (diagonal, horizontal, vertical)\n"
                "   --factorTransition=<f>     exponent of transition probabilities in log-linear combination\n"
                "   --model=<name>              name of the model to use. \n"
                "                                 zeroorder, conditional, simple\n"
                "   --order=<int>               order of the alignment model (does not apply to all models)\n"
                "   --lexiconFloor=<float>     floor value for the lexica\n"
                "   --lexiconType=<type>       lexicon type can be \"sri\" or \"plain\"\n"
                "   --outputXml=<filename>     output filename for xml format\n"
                "   --outputBilang=<filename>  output filename for bilanguage corpus\n"
                "   --outputAachen=<filename>  output filename for aachen format\n"
                "   --iterationsOrder=<intVector> iterations per order (default: 1,0. one iteration with zero order)\n"
                "   --outputLexicon=<filename>  output filename for lexicon\n"
                "   --sourceLmFsa=<filename>   source language model (to weight reorderings)"
                "\n"
                "translator options:\n"
                "   --beamPrune=<t>            prune output with beam threshold (default: infinity) (*)\n"
                "   --help                      print this page\n"
                "   --prune=<t>                 prune output with posterior threshold (default: infinity) (*)\n"
                "   --reorder.dfile=<file>      file with distortion probabilities \n"
                "   --reorder.distortion=<p>    distortion probability lambda (when no dfile is given, a parametric distribution will be used)\n"
                "   --reorder.max-distortion=<n>  maximum allowed distortion for ibm and inverse-ibm constraints (default: 20)\n"
                "   --reorder.probability=<p>   probability for main path (default: off)\n"
                "   --reorder.type=<n>          type of reordering (default: none)\n"
                "   --reorder.window-size=<n>   window size (default: infinity = full sentence)\n"
                "   --reorderlex=<file>         lexicon for nbest reorderings (*)\n"
                "   --factorReorder=<f>        exponent of the reordering model\n"
                "   --factorLexicon=<f>        exponent of the lexicon model\n"
                "   --factorSourceLm=<f>        exponent of the source languange model\n"
                "\n"
                "(*) under construction\n"
                "\n"
                "possible permutations:\n"
                "   ";
        std::ostringstream s;
        Translation::Reordering::typeChoice_.printIdentifiers(s);
        usage += s.str() + "\n\n";
        return usage;
}

Aligner::Aligner() : order_(0), reordering_(0),
                                         perplexity_(0.0), userTime_(0.0),
                                         sentences_(0), oovs_(0), words_(0), arcs_(0), maxMemory_(0), segmentStart_(0),
                                         segmentEnd_(0)
{
        setTitle("aligner");
        setDefaultLoadConfigurationFile(false);
        setDefaultOutputXmlHeader(false);
}

/**
 * Destructor giving some statistics about the translation that was performed
 */
Aligner::~Aligner() {
        std::cerr << "sentences: " << sentences_ << ". words: " << words_ << ".";
        if (words_) {
                // log probabilities in arpa format use logarithm to base 10 whereas we compare perplexities
                // usually based on natural logarithm
                std::cerr << " perplexity: " << ::exp((perplexity_ * ::log(10)) / words_) << ".";
                std::cerr << " " << oovs_ << " (= " << (100.0 * oovs_) / words_ << "%) oovs. "
                                  << arcs_ / words_ << " arcs/word.";
        }

        if (sentences_) std::cerr << " " << (1000.0 * userTime_) / sentences_ << " ms/sentence.";
        if (userTime_) std::cerr << " " << words_ / userTime_ << " words/sec.";
        if (maxMemory_) std::cerr << " max. memory: " << maxMemory_ / (1024.0 * 1024.0) << " MB.";
        std::cerr << std::endl;

        delete reordering_;

}

//extracts aligned words from a given linear automaton and stores them in a lexicon.
void Aligner::extract(Translation::ConditionalLexiconRef lexicon, ConstAutomatonRef f, int extractionOrder){

        Fsa::StateId initialState=f->initialStateId();
        Fsa::ConstStateRef currentState(f->getState(initialState));

        std::vector<std::string> sourceVector;
        std::vector<std::string> targetVector;
        size_t delta;
        std::vector<std::string> lexiconEntry(2*(extractionOrder+1),"<s>");

        while (!currentState->isFinal()) {


                totalCost_ = totalCost_ + float(currentState->begin()->weight());
//		std::cerr << "totalCost_: " << totalCost_ << std::endl;

                //delta 0
                if (currentState->begin()->output() == Epsilon) {

                        delta = 1;
                        sourceVector = Core::split(f->getInputAlphabet()->symbol(currentState->begin()->input()),"#");
                        targetVector = Core::split(f->getOutputAlphabet()->symbol(currentState->begin()->output()),"#");
                }

                //delta 2
                else if (currentState->begin()->input() == Epsilon) {

                        delta = 2;
                        sourceVector = Core::split(f->getInputAlphabet()->symbol(currentState->begin()->input()),"#");
                        targetVector = Core::split(f->getOutputAlphabet()->symbol(currentState->begin()->output()),"#");
                }

                //delta 1
                else {

                        delta = 0;
                        sourceVector = Core::split(f->getInputAlphabet()->symbol(currentState->begin()->input()),"#");
                        targetVector = Core::split(f->getOutputAlphabet()->symbol(currentState->begin()->output()),"#");
                }

                lexiconEntry.erase(lexiconEntry.begin());
                lexiconEntry.erase(lexiconEntry.begin());

                lexiconEntry.push_back(sourceVector[0]);
                lexiconEntry.push_back(targetVector[0]);

                lexicon->addValue(delta,lexiconEntry,1);
                currentState=f->getState(currentState->begin()->target());
        }
}


void Aligner::processFile(const std::string &sourceFilename,
                                                  const std::string &targetFilename) {



    // initialize the lexicon used to generate the alignments
        Translation::ConstConditionalLexiconRef conditionalLexicon;

    // this is confusing as it obvuscates the membership of the parameter lexicon.type
    // however, it was doen for consistency, so that all lexicon parameters are given with
    // the "lexicon" prefix
        switch (paramLexiconType_(select("lexicon"))) {
        case Translation::lexiconTypePlain :
                conditionalLexicon
                        = Translation::ConstConditionalLexiconRef(new Translation::ConditionalLexiconPlain(select("lexicon")));

                break;
        case Translation::lexiconTypeSri :
                conditionalLexicon
                        = Translation::ConstConditionalLexiconRef(new Translation::ConditionalLexiconSri(select("lexicon")));
                break;
        default :
                error("lexicon type unknown");
        }



        Translation::ConditionalLexiconRef tmpLexicon;

        // iterate through the iterationsOrder vector, which is given as a parameter.
        for (uint currentIteration=0; currentIteration<iterationsOrder_.size()-1; currentIteration=currentIteration+2 ) {
                // extraction the number of iterations per order out of the given iterationsOrder vector
                uint numberOfIterations = iterationsOrder_[currentIteration];
                uint extractionOrder = iterationsOrder_[currentIteration+1];
                uint nextExtractionOrder=0;

                if (currentIteration == iterationsOrder_.size()-2) {
                        nextExtractionOrder = extractionOrder;
                }
                else {
                        nextExtractionOrder = iterationsOrder_[currentIteration+3];
                }

                std::cerr << std::endl;

                // status information
                std::cerr << "extraction order: " << extractionOrder << std::endl;
                uint iteration = 0;
                order_ = extractionOrder;


                // begin iteration
                while (iteration< numberOfIterations) {

                std::cerr << "iteration : " << iteration+1;
                std::cerr << "(" <<  numberOfIterations << ")" << std::endl ;
        //	if (!writeLexicon_ && extractionOrder !=0 ) conditionalLexicon->write(std::cerr);

                // instantiate the lexicon for storing the extracted entries
                tmpLexicon = Translation::ConditionalLexiconRef(new Translation::ConditionalLexiconPlain(select("newlexicon")));

                Core::CompressedInputStream sourceStream(sourceFilename);
                Core::CompressedInputStream targetStream(targetFilename);

                Translation::Reordering* reordering = new Translation::Reordering(select("reorder"));

                // main loop over the source and target corpora
                while (sourceStream
                   && targetStream
                   && sentences_<=segmentEnd_
                ) {

                // read sentences
                        std::string sourceSentence;
                        std::string targetSentence;
                        Core::getline(sourceStream, sourceSentence);
                        Core::getline(targetStream, targetSentence);

                // process only if in given sentence range and everything is ok with the sentences
                        if (sourceStream
                                && targetStream
                                && sentences_>=segmentStart_
                                && sentences_<=segmentEnd_
                                && sourceSentence.size()>0
                                && targetSentence.size()>0
                    ) {

                    // status information
                                if (sentences_ % 1000 == 0) {
                                        std::cerr << "Sentence Number " << sentences_ << std::endl;
                                }

                    /*
                     * create the Alignment Automaton, that is a composition of
                     * - the reordered source sentence
                     * - the general alignment automaton constructed from the sentences and lexicon
                     *
                     * the resulting automaton contains ALL POSSIBLE alignments between source and target
                     */
                                ConstAutomatonRef alignment(createAlignment(sourceSentence,
                                                                                                                        targetSentence,
                                                                                                                        conditionalLexicon,
                                                                reordering));


                // search for best alignment or substitute by empty automaton, if alignment is broken
                    // nbest output could be included here
                                if (!isEmpty(alignment)) {
                                        alignment = best(cache(alignment));
                                } else {
                                        alignment = ConstAutomatonRef(staticCopy(std::string(" "), TropicalSemiring));
                                }

                    // counting of alignments should happen here
                        if (iteration == numberOfIterations-1) { //extraction with next order if at the end of the current iteration.
                                extract(tmpLexicon,alignment,nextExtractionOrder);
                        }
                        else {
                                extract(tmpLexicon,alignment,extractionOrder);
                        }


                    // write best alignment to output
                        if ((currentIteration == iterationsOrder_.size()-2) && ( iteration == numberOfIterations-1)){
                                if (writeXml_)    writeXml(alignment, outputXml_);
                                if (writeBiLang_) writeBiLang(alignment, outputBiLang_);
                                if (writeAachen_) writeAachen(alignment, outputAachen_, sentences_);
                        }
                        }
                        sentences_++;

                }

                sentences_ = segmentStart_ ;
                std::cerr << "totalCost: " << totalCost_ << std::endl;
                totalCost_ = 0;
                iteration++;
                // default normalization point 2
                tmpLexicon->normalize(normalizePoint_);

                // switch lexica after every iteration
                conditionalLexicon = tmpLexicon;

                }

                totalCost_ = 0;
        }
}

ConstAutomatonRef Aligner::createAlignment(std::string& sourceSentence,
                                           std::string& targetSentence,
                                           Translation::ConstConditionalLexiconRef conditionalLexicon,
                                           Translation::Reordering* reordering) {

        ConstAutomatonRef alignAutomaton(createAlignAutomaton(sourceSentence,
                                                                                                                  targetSentence,
                                                                                                                  conditionalLexicon));

        ConstAutomatonRef sourceAutomaton(staticCopy(numberTokens(sourceSentence), TropicalSemiring));

        sourceAutomaton = reordering->reorder(sourceAutomaton);
        sourceAutomaton = multiply(sourceAutomaton,factorReorder_);

        if (useSourceLm_) {
                sourceAutomaton = composeMatching(sourceAutomaton,sourceLmAutomaton_);
        }

        ConstAutomatonRef alignment =
                trim(
                        pruneSync(
                                composeMatching(
                                        sourceAutomaton,
                                        alignAutomaton
                                        ),
                                beamThreshold_
                                )
                        );

        return alignment;
}

ConstAutomatonRef Aligner::createAlignAutomaton(std::string& sourceSentence,
                                                                                                std::string& targetSentence,
                                                                                                Translation::ConstConditionalLexiconRef conditionalLexicon) {
        AlignAutomaton* aat(0);

//	switch (paramModel_(config)) {
//	case Fsa::modelZeroOrder :
//		aat = new ZeroOrderAlignAutomaton(config,
//										  sourceSentence,
//										  targetSentence,
//										  transitionProbs_,
//										  conditionalLexicon,
//										  factorLexicon_);
//		break;
//	case modelZeroOrderNoEmpty :
//		aat = new ZeroOrderNoEmptyAlignAutomaton(config,
//												 sourceSentence,
//												 targetSentence,
//												 transitionProbs_,
//												 lexicon,
//												 factorLexicon_);
//		break;
//	case modelConditional :
//		aat = new ConditionalAlignAutomaton(config,
//											sourceSentence,
//											targetSentence,
//											transitionProbs_,
//											conditionalLexicon,
//											factorLexicon_);
//		break;
//	case modelSimple :
                aat = new SimpleAlignAutomaton(config,
                                                                           sourceSentence,
                                                                           targetSentence,
                                                                           transitionProbs_,
                                                                           conditionalLexicon,
                                                                           factorLexicon_,
                                                                           factorTransition_,
                                                                           order_);
//		break;
//	default:
//		break;
//	}
        return ConstAutomatonRef(aat);
}

std::string Aligner::getConfiguration() {
    std::ostringstream oss;

    oss << "parameter settings:\n"
        << paramNBest_.name()                << " = " << paramNBest_(config) << "\n"
        << paramTransitionProbs_.name()      << " = " << Core::vector2String(paramTransitionProbs_(config)) << "\n"
        << paramFactorTransition_.name()     << " = " << paramFactorTransition_(config) << "\n"
        << paramFactorReorder_.name()        << " = " << paramFactorReorder_(config) << "\n"
        << paramFactorLexicon_.name()        << " = " << paramFactorLexicon_(config) << "\n"
                << paramFactorSourceLm_.name()   << " = " << paramFactorSourceLm_(config) << "\n"
        << paramModel_.name()                << " = " << paramModel_(config) << "\n"
        << paramSegmentStart_.name()         << " = " << paramSegmentStart_(config) << "\n"
        << paramSegmentEnd_.name()           << " = " << paramSegmentEnd_(config) << "\n"
        << paramPrune_.name()                << " = " << paramPrune_(config) << "\n"
        << paramBeamPrune_.name()            << " = " << paramBeamPrune_(config) << "\n"
        << paramOutputXmlFilename_.name()    << " = " << paramOutputXmlFilename_(config) << "\n"
        << paramOutputBiLangFilename_.name() << " = " << paramOutputBiLangFilename_(config) << "\n"
        << paramOutputAachenFilename_.name() << " = " << paramOutputAachenFilename_(config) << "\n"
        << paramOrder_.name()                << " = " << paramOrder_(config) << "\n"
                << paramSourceLmFilename_.name() << " = " << paramSourceLmFilename_(config) << "\n"
                ;
    return oss.str();
}

void Aligner::setMemberVariablesFromParameters() {
        // set translator-specific parameter variables as defined at the end of this file
        nbest_ = paramNBest_(config);
        order_ = paramOrder_(config);
        extractionOrder_ = paramExtractionOrder_(config);
        normalize_ = paramNormalize_(config);
        normalizePoint_ = paramNormalizePoint_(config);
        iterations_ = paramIterations_(config);
        writeXml_ = paramOutputXmlFilename_(config) != "";
        writeBiLang_ = paramOutputBiLangFilename_(config) != "";
        writeAachen_ = paramOutputAachenFilename_(config) != "";
        writeLexicon_ = paramOutputLexiconFilename_(config) != "";
        factorReorder_ = Fsa::Weight(paramFactorReorder_(config));
        factorTransition_ = Fsa::Weight(paramFactorTransition_(config));
        factorLexicon_ = Fsa::Weight(paramFactorLexicon_(config));
        factorSourceLm_ = Fsa::Weight(paramFactorSourceLm_(config));
        sourceLmFilename_ = paramSourceLmFilename_(config);
        totalCost_ = Fsa::Weight(0);

        transitionProbs_=paramTransitionProbs_(config);
        if (transitionProbs_.size()<3) {
                transitionProbs_.resize(3);
                for (size_t i=0;i<3;++i) {
            transitionProbs_[i]=3;
                }
        }


        iterationsOrder_=paramIterationsOrder_(config);

        Core::negLogVector(transitionProbs_);

        std::string threshold = paramPrune_(config);
        if (threshold == "") threshold_ = TropicalSemiring->max();
        else threshold_ = TropicalSemiring->fromString(threshold);
        threshold = paramBeamPrune_(config);
        if (threshold == "") beamThreshold_ = TropicalSemiring->max();
        else beamThreshold_ = TropicalSemiring->fromString(threshold);

        //floor_ = paramFloor_(config);
        segmentStart_ = paramSegmentStart_(config);
        segmentEnd_ = paramSegmentEnd_(config);

        useSourceLm_ = (sourceLmFilename_ != "");


}


int Aligner::main(const std::vector<std::string> &arguments) {
        // initialize translation automaton

         if (arguments.size()<2) {
                std::cerr << getUsage();
                return 1;
        }

    std::cerr << "source: " << arguments[0] << "\n"
              << "target: " << arguments[1] << "\n";

    setMemberVariablesFromParameters();

        if (writeXml_) {
                outputXml_.open(paramOutputXmlFilename_(config));
                writeXml_ = outputXml_.good();
        }

        if (writeBiLang_) {
                outputBiLang_.open(paramOutputBiLangFilename_(config));
                writeBiLang_ = outputBiLang_.good();
        }

        if (writeAachen_) {
                outputAachen_.open(paramOutputAachenFilename_(config));
                writeAachen_ = outputAachen_.good();
        }

        if (writeLexicon_) {
                outputLexicon_.open(paramOutputLexiconFilename_(config));
                writeLexicon_ = outputLexicon_.good();
        }

        std::cerr << getConfiguration();

        if (useSourceLm_) {
                Fsa::StaticAutomaton lm;
                sourceLmAutomaton_ = Fsa::read(sourceLmFilename_, Fsa::TropicalSemiring);
        }

        processFile(arguments[0], arguments[1]);

        return EXIT_SUCCESS;
}


Core::ParameterChoice Aligner::paramLexiconType_(
        "type", &Translation::lexiconTypeChoice, "lexicon type can be sri or plain", Translation::lexiconTypePlain);

// I/O Parameters
Core::ParameterInt Aligner::paramSegmentStart_(
    "startSentence", "sentence to start with", 0, 0);
Core::ParameterInt Aligner::paramSegmentEnd_(
    "endSentence", "sentence to stop at", 1000000, 0);
Core::ParameterString Aligner::paramOutputXmlFilename_(
         "outputXml", "output file for \"xml\" alignment format", "");
Core::ParameterString Aligner::paramOutputBiLangFilename_(
         "outputBilang", "output file for bilanguage corpus", "");
Core::ParameterString Aligner::paramOutputAachenFilename_(
         "outputAachen", "output file for aachen alignment format", "");
Core::ParameterInt Aligner::paramNBest_(
   "nBest", "generate n-best list instead of single best", 0);
Core::ParameterString Aligner::paramOutputLexiconFilename_(
         "outputLexicon", "output file for lexicon", "");
Core::ParameterString Aligner::paramSourceLmFilename_(
         "sourceLm", "source language model filename (fsa format)", "");
Core::ParameterBool Aligner::paramNormalize_(
   "normalize", "normalization");
Core::ParameterInt Aligner::paramNormalizePoint_(
   "normalizePoint", "normalization point", 2);
Core::ParameterInt Aligner::paramIterations_(
   "iterations", "number of iterations", 1);


// model parameters
Core::ParameterChoice Aligner::paramModel_(
        "model", &modelChoice, "model to use for alignment", modelSimple);
Core::ParameterInt Aligner::paramOrder_(
    "order", "order of the model", 0, 0);
Core::ParameterInt Aligner::paramExtractionOrder_(
    "extractionOrder", "order of the extraction", 0, 0);
Core::ParameterFloatVector Aligner::paramTransitionProbs_(
         "transitionProbs", "list of probabilites for the transitions permitted in the model", ",");
Core::ParameterFloat Aligner::paramFactorTransition_(
    "factorTransition", "factor for transition probabilities", 1.0);
Core::ParameterFloat Aligner::paramFactorReorder_(
    "factorReorder", "factor for reordering probabilities", 1.0);
Core::ParameterFloat Aligner::paramFactorLexicon_(
    "factorLexicon", "factor for Lexicon probabilities", 1.0);
Core::ParameterFloat Aligner::paramFactorSourceLm_(
    "factorSourceLm", "factor for source language model", 1.0);
Core::ParameterIntVector Aligner::paramIterationsOrder_(
         "iterationsOrder", "list of iterations,order ", ",");
// Search Parameters
Core::ParameterString Aligner::paramPrune_(
    "posteriorPrune", "prune output with threshold (posterior probabilities)", "");
Core::ParameterString Aligner::paramBeamPrune_(
    "beamPrune", "prune output with beam threshold", "");
