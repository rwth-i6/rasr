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
// $Id$

#include <Bliss/Lexicon.hh>
#include <Core/Application.hh>
#include <Core/Types.hh>
#include <Core/Version.hh>
#include <Legacy/DecisionTree.hh>
#include <Legacy/MeanSumfileEstimator.hh>
#include <Legacy/MixtureSet.hh>
#include <Mm/MixtureSet.hh>

class MeanSumfileCreator : public virtual Core::Application {
    static const Core::ParameterString paramMixtureSetFilename;
    static const Core::ParameterString paramSumfileFilename;

public:
    virtual string getUsage() const {
        return "build sumfile for MLLR regression class tree estimation";
    }

    MeanSumfileCreator()
            : Core::Application() {
        setTitle("mean-sumfile-creator");
    }

    int main(const vector<string>& arguments);
};

APPLICATION(MeanSumfileCreator)

const Core::ParameterString MeanSumfileCreator::paramMixtureSetFilename(
        "mixture-set-file",
        "name of (legacy) reference file to load");

const Core::ParameterString MeanSumfileCreator::paramSumfileFilename(
        "sumfile",
        "name of sumfile for MLLR regression class tree estimation");

int MeanSumfileCreator::main(const vector<string>& arguments) {
    select("lexicon");

    Bliss::LexiconRef lexicon = Bliss::Lexicon::create(select("lexicon"));
    if (!lexicon)
        criticalError("failed to initialize lexicon");

    const Legacy::PhoneticDecisionTree* dectree = new Legacy::PhoneticDecisionTree(select("decision-tree"),
                                                                                   lexicon->phonemeInventory());
    dectree->respondToDelayedErrors();

    Core::Ref<Legacy::MixtureSet> mixtureSet = Legacy::createMixtureSet(paramMixtureSetFilename(config));
    if (!mixtureSet)
        criticalError("failed to load mixture set");

    std::vector<std::set<u32>>      phonemeToMixtureIndizes = dectree->PhonemeToMixtureIndizes();
    std::vector<Bliss::Phoneme::Id> mixtureToPhoneme(mixtureSet->nMixtures(), -1);
    for (Bliss::Phoneme::Id p = 1; p < phonemeToMixtureIndizes.size(); ++p) {
        for (std::set<u32>::const_iterator m = phonemeToMixtureIndizes[p].begin(); m != phonemeToMixtureIndizes[p].end(); ++m) {
            mixtureToPhoneme[*m] = p;
        }
    }

    for (std::vector<Bliss::Phoneme::Id>::const_iterator m = mixtureToPhoneme.begin();
         m != mixtureToPhoneme.end(); ++m)
        if (*m == -1)
            error("no entry in mixtureToPhoneme for mixture ") << *m;

    Legacy::MeanSumfileEstimator meanSumfileEstimator(mixtureSet, mixtureToPhoneme, lexicon->phonemeInventory());

    std::string test(paramSumfileFilename(config));
    meanSumfileEstimator.write(test);

    delete dectree;
    return 0;
}
