#include "SearchV2.hh"

namespace Search {
SearchAlgorithmV2::SearchAlgorithmV2(const Core::Configuration& config)
        : Core::Component(config) {}

SearchAlgorithmV2::Score SearchAlgorithmV2::ScoreMap::sum() const {
    Score scoreSum(0.0);
    for (auto it = begin(); it != end(); ++it) {
        scoreSum += it->second;
    }
    return scoreSum;
}

SearchAlgorithmV2::ScoreMap SearchAlgorithmV2::ScoreMap::operator+(ScoreMap const& other) const {
    ScoreMap result = *this;
    for (const auto& [key, value] : other) {
        result.at(key) += value;
    }
    return result;
}

SearchAlgorithmV2::ScoreMap SearchAlgorithmV2::ScoreMap::operator-(ScoreMap const& other) const {
    ScoreMap result = *this;
    for (const auto& [key, value] : other) {
        result.at(key) -= value;
    }
    return result;
}

SearchAlgorithmV2::ScoreMap SearchAlgorithmV2::ScoreMap::operator+=(ScoreMap const& other) {
    for (const auto& [key, value] : other) {
        at(key) = value;
    }
    return *this;
}

SearchAlgorithmV2::ScoreMap SearchAlgorithmV2::ScoreMap::operator-=(ScoreMap const& other) {
    for (const auto& [key, value] : other) {
        at(key) -= value;
    }
    return *this;
}

}  // namespace Search
