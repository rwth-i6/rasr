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
#include "Random.hh"
#ifndef CMAKE_DISABLE_MODULE_HH
#include <Modules.hh>
#endif
#ifdef MODULE_MATH_NR
#include "Nr/Random.hh"
#endif
#include <Core/Assertions.hh>
#include <libxml/parser.h>

namespace Math {

std::mt19937 randomEngine;

void randomSeed(long seed) {
    // Init our own random number generator.
    // This should be the base number generator for all our internal random numbers.
    randomEngine.seed((uint32_t)seed);

    // On the first call to xmlDictCreate,
    // libxml2 will initialize some internal randomize system,
    // which calls srand(time(NULL)).
    // So, do that first call here now, so that we can use our
    // own random seed.
    xmlDictPtr p = xmlDictCreate();
    xmlDictFree(p);

    // Now we can set the seed on the global libc random number generator.
    // Let's hope that no-one else resets that at some later time.
    // Use some random number as the seed.
    srand((unsigned int)randomEngine());

    // Note: We should not use the C/C++ global random
    // generator, because every other lib could call `srand()`.
    // We must use our own global random generator.
    // libxml2 also wont change that and even states that the code
    // above is the "correct" way to do it: https://bugzilla.gnome.org/show_bug.cgi?id=738231
    // Some discussion: http://stackoverflow.com/questions/26294162/usefullness-of-rand-or-who-should-call-srand
}

static std::uniform_int_distribution<> randIntUniform(0, RAND_MAX);

// Return a random integer between 0 and RAND_MAX inclusive.
int rand() {
    return (int)randIntUniform(randomEngine);
}

#ifdef MODULE_MATH_NR

const Core::Choice RandomVectorGenerator::choiceType(
        "uniform-independent", typeUniformIndependent,
        "gauss-independent", typeGaussIndependent,
        Core::Choice::endMark());
const Core::ParameterChoice RandomVectorGenerator::paramType(
        "type", &choiceType, "type of distribution", typeUniformIndependent);

RandomVectorGenerator* RandomVectorGenerator::create(Type type) {
    switch (type) {
        case typeUniformIndependent: return new IndependentRandomVectorGenerator<Nr::Ran2>;
        case typeGaussIndependent: return new IndependentRandomVectorGenerator<Nr::Gasdev<Nr::Ran2>>;
    }
    defect();
    return 0;
}

#endif

}  // namespace Math
