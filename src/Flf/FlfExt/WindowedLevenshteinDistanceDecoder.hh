#ifndef _FLF_WINDOWED_LEVENSHTEIN_DISTANCE_DECODER_HH
#define _FLF_WINDOWED_LEVENSHTEIN_DISTANCE_DECODER_HH

/**
   Implementation of the approximate Bayes risk decoder with windowed Levenshtein distance as loss function.
   The windowed Levenshtein distance is centered around an initial CN alignment.

   For details see my thesis, chapter 5.2.2 (From CN Decoding to Bayes risk Decoding ...)
**/

#include <Flf/FlfCore/Lattice.hh>
#include <Flf/FwdBwd.hh>
#include <Flf/Network.hh>

/**
 * Trace alignment
 **/
//define WINDOWED_LEVENSHTEIN_DECODER_FULL_ALIGNMENT

namespace Flf {

/*
          Use lattice fwd/bwd scores together with a CN to compute conditional word posteriors
        */
class ConditionalPosterior;
typedef Core::Ref<const ConditionalPosterior> ConstConditionalPosteriorRef;

class ConditionalPosterior : public Core::ReferenceCounted {
public:
    struct Value {
        Fsa::LabelId label;                // x_n
        f64          condPosteriorScore;   // P(x_n | x_1 ... x_{n-1})
        f64          tuplePosteriorScore;  // P(x_1 ... x_n)
        Value(Fsa::LabelId label, Score condPosteriorScore, Score tuplePosteriorScore);
    };
    typedef std::vector<Value>                      ValueList;
    typedef ValueList::const_iterator               ValueIterator;
    typedef std::pair<ValueIterator, ValueIterator> ValueRange;

public:
    class Internal;

private:
    Internal* internal_;

private:
    ConditionalPosterior(Internal* internal);

public:
    ~ConditionalPosterior();

    u32 contextSize() const;

    void dump(std::ostream& os) const;

    /*
     * return ( -log(P_{position}(labels[-1]| labels[0:-1]), -log(P_{position, position-len(labels)}(labels) )
     * - len(labels) has to equal contextLength() + 1
     * - labels[0:len(labels)-position-1] are ignored, i.e. supposed to be epsilons
     */
    const Value& posterior(u32 position, const LabelIdList& labels) const;
    ValueRange   posteriors(u32 position, const LabelIdList& labels) const;

    static ConstConditionalPosteriorRef create(
            ConstLatticeRef l, ConstFwdBwdRef fb, ConstConfusionNetworkRef cn, u32 contextSize, bool compact = true);
};

/*
 * Compute conditional posteriors from lattice
 */
NodeRef createConditionalPosteriorsNode(const std::string& name, const Core::Configuration& config);

/*
 * Windowed Levenshtein distance deocder
 */
NodeRef createWindowedLevenshteinDistanceDecoderNode(const std::string& name, const Core::Configuration& config);

}  // namespace Flf

#endif  // _FLF_WINDOWED_LEVENSHTEIN_DISTANCE_DECODER_HH
