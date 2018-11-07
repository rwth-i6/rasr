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
#include <Core/Application.hh>
#include <Core/BinaryStream.hh>

#include "MixtureSet.hh"
#include "MixtureSetEstimator.hh"
#include "FeatureScorer.hh"
#include "Module.hh"

class TestApplication : public Core::Application {
public:
    TestApplication() : Core::Application() {
        INIT_MODULE(Mm)
        setTitle("check");
    }
    virtual std::string getUsage() const { return "short program to test Mm features\n"; }
    int main(const std::vector<std::string> &arguments);
};

/*****************************************************************************/
int TestApplication::main(const std::vector<std::string> &arguments)
/*****************************************************************************/
{
#if 0
    int N_mix=2501, N_cmp_total=33;
    std::cout << "---" << std::endl;
    //Mm::MixtureSetWithVariance<> *tmp = new Mm::MixtureSetWithVariance<>(N_mix+1, N_cmp_total, 0.15 * N_cmp_total);
    std::cout << "---" << std::endl;
    Core::Ref<Mm0::MixtureSetWithVariance<> > mixtures(new Mm0::MixtureSetWithVariance<>(N_mix+1, N_cmp_total, 0.15 * N_cmp_total));

    Mm0::FeatureVector featureVector(N_cmp_total,0.0);

    Mm::FeatureScorer * fs;
    Mm::FeatureScorer::Scorer scorer;

    fs = new Mm0::FeatureScorerGaussViterbi<Mm::MixtureSetWithVariance<> >(select("xxx"), mixtures);

    std::string filename = "OldRef.ref";
    Core::Ref<ClassicMixtureSet > legacyMixtureSet;
    Mm::Application::us()->readMixtureSet(filename, legacyMixtureSet);

    scorer = fs->get(featureVector);

    mixtures->calculateNormFactor();

    std::ifstream refistrm(filename.c_str(),std::ios::in|std::ios::binary);
    std::ofstream refostrm(filename.c_str(),std::ios::out|std::ios::binary);

    mixtures->write(refostrm);
    mixtures->read(refistrm);
#endif

#if 0
    Mm0::MixtureSetPooledVarEstimator<> estimator(2501, 33);
    std::ofstream o("test.ref");
    Core::BinaryOutputStream os(o);
    estimator.write(os);

    std::ifstream i("test.ref");
    Core::BinaryInputStream is(i);
    estimator.read(is);
#endif
    Mm::Module::instance().createFeatureScorer(config);
    return 0;
}

APPLICATION(TestApplication)
