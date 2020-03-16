#ifndef _LM_CHEATING_SEGMENT_LM_HH
#define _LM_CHEATING_SEGMENT_LM_HH

#include <Bliss/Orthography.hh>

#include "FsaLm.hh"

namespace Lm {

class CheatingSegmentLm : public FsaLm {
public:
    using Precursor = FsaLm;

    static const Core::ParameterFloat paramInfinityScore;

    CheatingSegmentLm(Core::Configuration const& c, Core::Ref<const Bliss::Lexicon> l);
    virtual ~CheatingSegmentLm() = default;

    virtual void load();
    virtual void setSegment(Bliss::SpeechSegment const* s);
private:
    Core::Ref<const Bliss::Lexicon>            lexicon_;
    Core::Ref<const Bliss::OrthographicParser> orthParser_;
};

// inline implementation

inline CheatingSegmentLm::CheatingSegmentLm(Core::Configuration const& c, Bliss::LexiconRef l)
    : Core::Component(c), Precursor(c, l), lexicon_(l), orthParser_(Core::ref(new Bliss::OrthographicParser(config, lexicon_))) {
    infinityScore_ = paramInfinityScore(config);
}

}  // namespace Lm

#endif  // _LM_CHEATING_SEGMENT_LM_HH
