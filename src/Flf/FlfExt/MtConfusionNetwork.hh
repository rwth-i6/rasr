#ifndef _FLF_MT_CONFUSION_NETWORK_HH
#define _FLF_MT_CONFUSION_NETWORK_HH

/**
   Implementation to enriche lattices with information from a CN.
   Used in machine translation by
   - Evgeny Matusov: "ASR Word Lattice Translation with Exhaustive Reordering is Possible", Interspeech 2008
   - Yuqi Zhang

   Remark: slightly hackish; shall be removed, if not needed by MT anymore, or overhauled if there is a regular need
**/

#include <Flf/FlfCore/Lattice.hh>
#include <Flf/Network.hh>

namespace Flf {

NodeRef createMtCnFeatureNode(const std::string& name, const Core::Configuration& config);

NodeRef createMtNormalizedCnPruningNode(const std::string& name, const Core::Configuration& config);

}  // namespace Flf

#endif  // _FLF_MT_CONFUSION_NETWORK_HH
