#ifndef _FLF_MAP_DECODER_HH
#define _FLF_MAP_DECODER_HH

/**
   Implementation of sentence error based decoder for single and multiple lattice input:
   - Viterbi vs. MAP, word boundaries are computed from frame-wise word posteriors, see <Flf/TimeAlignment.hh>
   - Intersection vs. Union

   For details see my thesis, chapter 3.3.1 (The MAP/Viterbi Decoding Framework)
**/

#include <Flf/FlfCore/Lattice.hh>
#include <Flf/Network.hh>

namespace Flf {

NodeRef createMapDecoderNode(const std::string& name, const Core::Configuration& config);
NodeRef createIntersectionMapDecoderNode(const std::string& name, const Core::Configuration& config);
NodeRef createUnionMapDecoderNode(const std::string& name, const Core::Configuration& config);

}  // namespace Flf

#endif  // _FLF_MAP_DECODER_HH
