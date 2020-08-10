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
#include <Core/Application.hh>

class FeatureStatistics : public Core::Application {
public:
    enum Action {
        actionDry,
        actionApplyScatterMatrixThreshold,
        actionEstimateHistograms,
        actionEstimateMean,
        actionEstimateCovariance,
        actionEstimatePca,
        actionEstimateCovarianceAndPca,
        actionCalculateCovarianceDiagonalNormalization,
        actionNotGiven
    };

    static const Core::Choice          choiceAction;
    static const Core::ParameterChoice paramAction;

private:
    void dryRun();
    void applyScatterMatrixThreshold();
    void estimateHistograms();
    void estimateMean();
    void estimateCovariance();
    void estimatePca();
    void estimateCovarianceAndPca();
    void calculateCovarianceDiagonalNormalization();

    void visitCorpus(Speech::CorpusProcessor& corpusProcessor);

public:
    FeatureStatistics();
    ~FeatureStatistics() {}

    virtual std::string getUsage() const {
        return "Creates statistics over the extracted features";
    }
    virtual int main(const std::vector<std::string>& arguments);
};
