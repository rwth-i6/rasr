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
#include <Am/ClassicAcousticModel.hh>
#include <Am/ClassicHmmTopologySet.hh>
#include <Bliss/Phoneme.hh>
#include <Cart/Example.hh>
// include <Cart/Parser.hh>
#include <Core/Parameter.hh>
#include <Core/XmlStream.hh>

#include <Core/OpenMPWrapper.hh>
#include "DecisionTreeTrainer.hh"

using namespace Speech;

// ============================================================================
FeatureAccumulator::FeatureAccumulator(
        const Core::Configuration& config)
        : Core::Component(config),
          Precursor(config, Am::AcousticModel::noEmissions | Am::AcousticModel::noStateTying),
          examples_(config),
          nObs_(0),
          nCols_(0),
          map_() {
    map_ = Cart::PropertyMapRef(
            new Am::PropertyMap(
                    (dynamic_cast<const Am::ClassicAcousticModel&>(*acousticModel())).stateModel()));
    examples_.setMap(map_);
}

Cart::Example& FeatureAccumulator::example(Am::AllophoneStateIndex id) {
    if ((size_t(id) >= examples_.size()) || !examples_[id]) {
        Am::Properties props(map_, allophoneState(id));
        Cart::Example* _example = new Cart::Example(new Cart::StoredProperties(props), new Cart::FloatBox(2, nCols_));
        examples_.set(id, _example);
        return *_example;
    }
    else
        return *examples_[id];
}

void FeatureAccumulator::processAlignedFeature(Core::Ref<const Feature> f, Am::AllophoneStateIndex id) {
    processAlignedFeature(f, id, 1.0);
}

void FeatureAccumulator::processAlignedFeature(Core::Ref<const Feature> f, Am::AllophoneStateIndex id, Mm::Weight w) {
    if (w == 0.0)
        return;
    if (acousticModel()->allophoneStateAlphabet()->isDisambiguator(id)) {
        warning("Disambiguator %s found in alignment",
                acousticModel()->allophoneStateAlphabet()->symbol(id).c_str());
        return;
    }
    const Mm::FeatureVector& feature = *f->mainStream();
    // Do I see my first feature? Yes, then initialize example size
    if (nCols_)
        verify(nCols_ == feature.size());
    else {
        nCols_ = feature.size();
        require(nCols_ > 0);
    }
    nObs_ += w;
    if ((size_t(id) >= examples_.size()) || !examples_[id]) {
        Am::Properties props(map_, acousticModel()->allophoneStateAlphabet()->allophoneState(id));
        examples_.set(id, new Cart::Example(new Cart::StoredProperties(props), new Cart::FloatBox(2, nCols_)));
    }
    Cart::Example& example = *examples_[id];
    //    example.write((Core::XmlWriter&)log());

    Mm::FeatureVector::const_iterator itFeature      = feature.begin();
    Cart::FloatBox::vector_iterator   itSum          = example.values->row(0).begin();
    Cart::FloatBox::vector_iterator   itSumOfSquares = example.values->row(1).begin();
    for (; itFeature != feature.end(); ++itFeature, ++itSum, ++itSumOfSquares) {
        *itSum += (*itFeature) * w;
        *itSumOfSquares += ((*itFeature) * (*itFeature)) * w;
    }
    example.nObs += w;
}

void FeatureAccumulator::write(std::ostream& out) const {
    out << "#examples     : " << examples_.size() << std::endl
        << "#observations : " << nObs_ << std::endl
        << "matrix size   : " << nCols_ << " x " << 2 << std::endl;
}

void FeatureAccumulator::writeXml(Core::XmlWriter& xml) const {
    xml << Core::XmlFull("nExamples", examples_.size())
        << Core::XmlFull("nObs", nObs_)
        << Core::XmlEmpty("matrix-f64") + Core::XmlAttribute("nRows", 2) + Core::XmlAttribute("nColumns", nCols_);
}
// ============================================================================

// ============================================================================
#define PI 3.141592653589793238462643383279502884197169399375105820975E+0000

const Core::ParameterFloat StateTyingDecisionTreeTrainer::LogLikelihoodGain::paramVarianceClipping(
        "variance-clipping",
        "minimum \\sigma^2",
        0.0);

StateTyingDecisionTreeTrainer::LogLikelihoodGain::LogLikelihoodGain(const Core::Configuration& config)
        : Precursor(config) {
    minSigmaSquare_ = paramVarianceClipping(config);
    parallel_       = paramDoParallel(config);
}

void StateTyingDecisionTreeTrainer::LogLikelihoodGain::write(std::ostream& os) const {
    os << "log-likelihood-gain\n"
       << "variance-clipping: " << minSigmaSquare_ << "\n";
}

/*
  negative log-likelihood for a Gaussian with diagonal co-variance matrix

  for x_1^N with x_n = [ x_{n,1}, ... x_{n,D} ]
  \mu_d              = \frac{1}{N} \sum_{n=1}^{N} x_{n,d}
  \sigma^2_d         = \frac{1}{N} \sum_{n=1}^{N} (x_{n,d} - \mu_{n,d})^2
  \theta             = ([\mu]_1^D, [\sigma^2]_^D)
  log N(x; \theta)   = -\frac{1}{2} \left( \sum_1^D \log(2 \pi \sigma^2_d + \sum_1^D \frac{(x_d - \mu_d)^2}{\sigma^2_d} \right)
  -LL(\theta| x_1^N) = \frac{1}{2} \left( N D + N \sum_{d=1}^{D} \log( 2 \pi \sigma^2_d) \right)
*/
Cart::Score StateTyingDecisionTreeTrainer::LogLikelihoodGain::logLikelihood(
        Cart::ExamplePtrList::const_iterator begin,
        Cart::ExamplePtrList::const_iterator end) const {
    if (begin == end)
        return 0.0;

    size_t d = (*begin)->values->columns();
    mu_.resize(d, 0.0);
    sigmaSquare_.resize(d, 0.0);
    f64 count = 0;
    f64 ll    = 0.0;

    if (parallel_) {
#pragma omp parallel
        {
            std::vector<f64> mu_private(d, 0.0);           // to store temporary results
            std::vector<f64> sigmaSquare_private(d, 0.0);  // to store temporary results

#pragma omp for nowait reduction(+ : count)
            for (Cart::ExamplePtrList::const_iterator it = begin; it < end; ++it) {  // supported by omp 3.0
                const Cart::Example& example = **it;
                require(example.values->rows() == 2);
                require(example.values->columns() == d);

                count += example.nObs;

                std::vector<f64>::iterator            itMu           = mu_private.begin();
                std::vector<f64>::iterator            itSigmaSquare  = sigmaSquare_private.begin();
                Cart::FloatBox::const_vector_iterator itSum          = example.values->row(0).begin();
                Cart::FloatBox::const_vector_iterator itSumOfSquares = example.values->row(1).begin();
                for (; itMu != mu_private.end(); ++itMu, ++itSigmaSquare, ++itSum, ++itSumOfSquares) {
                    *itMu += *itSum;
                    *itSigmaSquare += *itSumOfSquares;
                }
            }

#pragma omp critical
            {
                /**
                 * if slow create e.g. std::vector<std::vector<f64>> mus_private(threadnum, std::vector<f64>(d, 0.0)) and merge them
                 * with an 'omp for' over dimension and also merge with next for loop;
                 * order is not defined, non-deterministic sum-up!
                 */
                std::vector<f64>::iterator itMu_private          = mu_private.begin();
                std::vector<f64>::iterator itSigmaSquare_private = sigmaSquare_private.begin();
                std::vector<f64>::iterator itMu                  = mu_.begin();
                std::vector<f64>::iterator itSigmaSquare         = sigmaSquare_.begin();
                for (; itMu != mu_.end(); ++itMu, ++itSigmaSquare, ++itMu_private, ++itSigmaSquare_private) {
                    *itMu += *itMu_private;
                    *itSigmaSquare += *itSigmaSquare_private;
                }
            }

#pragma omp barrier  // each thread should have access to count, mu, sigmsq
#pragma omp for reduction(+ : ll)
            for (size_t dimCounter = 0; dimCounter < d; ++dimCounter) {
                mu_[dimCounter] /= count;
                sigmaSquare_[dimCounter] /= count;
                sigmaSquare_[dimCounter] -= mu_[dimCounter] * mu_[dimCounter];
                if (sigmaSquare_[dimCounter] < minSigmaSquare_)
                    sigmaSquare_[dimCounter] = minSigmaSquare_;

                ll += ::log(sigmaSquare_[dimCounter]);

                mu_[dimCounter] = sigmaSquare_[dimCounter] = 0.0;
            }
        }
        ll = (0.5 * count) * (d + d * ::log(PI + PI) + ll);
        return ll;
    }
    else {  // backward compatibility
        for (Cart::ExamplePtrList::const_iterator it = begin; it != end; ++it) {
            const Cart::Example& example = **it;
            require(example.values->rows() == 2);
            require(example.values->columns() == d);
            count += example.nObs;
            std::vector<f64>::iterator            itMu           = mu_.begin();
            std::vector<f64>::iterator            itSigmaSquare  = sigmaSquare_.begin();
            Cart::FloatBox::const_vector_iterator itSum          = example.values->row(0).begin();
            Cart::FloatBox::const_vector_iterator itSumOfSquares = example.values->row(1).begin();
            for (; itMu != mu_.end(); ++itMu, ++itSigmaSquare, ++itSum, ++itSumOfSquares) {
                *itMu += *itSum;
                *itSigmaSquare += *itSumOfSquares;
            }
        }
        f64                        n             = count;
        std::vector<f64>::iterator itMu          = mu_.begin();
        std::vector<f64>::iterator itSigmaSquare = sigmaSquare_.begin();
        for (; itMu != mu_.end(); ++itMu, ++itSigmaSquare) {
            *itMu /= n;
            *itSigmaSquare /= n;
            *itSigmaSquare -= (*itMu) * (*itMu);
            if (*itSigmaSquare < minSigmaSquare_)
                *itSigmaSquare = minSigmaSquare_;
            ll += ::log(*itSigmaSquare);
            *itMu = *itSigmaSquare = 0.0;
        }
        ll = (0.5 * n) * (d + d * ::log(PI + PI) + ll);
        return ll;
    }
}

void StateTyingDecisionTreeTrainer::LogLikelihoodGain::operator()(
        const Cart::ExamplePtrRange& examples,
        Cart::Score&                 score) const {
    score = logLikelihood(examples.begin, examples.end);
}

Cart::Score StateTyingDecisionTreeTrainer::LogLikelihoodGain::operator()(
        const Cart::ExamplePtrRange& leftExamples, const Cart::ExamplePtrRange& rightExamples,
        const Cart::Score fatherLogLikelihood,
        Cart::Score& leftChildLogLikelihood, Cart::Score& rightChildLogLikelihood) const {
    leftChildLogLikelihood  = logLikelihood(leftExamples.begin, leftExamples.end);
    rightChildLogLikelihood = logLikelihood(rightExamples.begin, rightExamples.end);
    Cart::Score gain        = fatherLogLikelihood - (leftChildLogLikelihood + rightChildLogLikelihood);
    return gain;
}

StateTyingDecisionTreeTrainer::StateTyingDecisionTreeTrainer(const Core::Configuration& config)
        : Precursor(config) {
    setScorer(Cart::ConstScorerRef(new LogLikelihoodGain(select("log-likelihood-gain"))));
}

const Core::ParameterBool StateTyingDecisionTreeTrainer::LogLikelihoodGain::paramDoParallel(
        "cluster-parallel",
        "use OpenMP (e.g. OMP_NUM_THREADS environment value) to parallelize score calculation; due to backward compatibility explicit flag", false);

// ============================================================================
