#include "SearchV2.hh"

namespace Search {
SearchAlgorithmV2::SearchAlgorithmV2(const Core::Configuration& config)
        : Core::Component(config) {}

bool SearchAlgorithmV2::decodeMore() {
    bool success = false;
    while (decodeStep()) {
        success = true;
    }
    return success;
}

}  // namespace Search
