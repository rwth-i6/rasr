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
#include <Search/Wfst/LatticeNetwork.hh>
#include <Search/Wfst/GrammarFst.hh>
#include <Search/Wfst/ComposedNetwork.hh>
#include <Search/Wfst/Lattice.hh>
#include <Search/Wfst/LatticeArchive.hh>
#include <Search/Wfst/LexiconFst.hh>
#include <fst/lookahead-matcher.h>

using namespace Search::Wfst;

const Core::ParameterString LatticeNetwork::paramLexiconFst(
    "lexicon-fst", "lexicon fst", "", "L to be composed with every lattice");
const Core::ParameterString LatticeNetwork::paramLmFst(
    "lm-fst", "LM fst", "", "G to be composed with every lattice");
const Core::ParameterBool LatticeNetwork::paramRemoveLexiconWeights(
    "remove-weights-l", "remove weights from L", true);

LatticeNetwork::~LatticeNetwork()
{
    delete l_;
    delete g_;
    delete archive_;
}

bool LatticeNetwork::init()
{
    log("using lattice re-scoring network");
    delete archive_;
    archive_ = new LatticeArchive(select("lattice-archive"));
    const std::string lFile = paramLexiconFst(config);
    const std::string gFile = paramLmFst(config);
    if (!gFile.empty() && !loadGrammar(gFile)) return false;
    if (!lFile.empty() && !loadLexicon(lFile)) return false;
    f_ = new OpenFst::VectorFst;
    return true;
}

bool LatticeNetwork::loadLexicon(const std::string &file)
{
    delete l_;
    LexicalFstFactory::Options options;
    // options.grammarType = AbstractGrammarFst::TypeVector;
    options.accumulatorType = LexicalFstFactory::DefaultAccumulator;
    options.lookAhead = LabelLookAhead;
    log("loading lexicon fst: %s", file.c_str());
    LexicalFstFactory factory(select("lexicon-fst"));
    if (paramRemoveLexiconWeights(config)) {
        FstLib::StdVectorFst *base = FstLib::StdVectorFst::Read(file);
        log("removing lexicon weights");
        FstLib::ArcMap(base, FstLib::RmWeightMapper<OpenFst::Arc>());
        l_ = factory.convert(base, options, 0);
        delete base;
    } else {
        l_ = factory.load(file, options, 0);
    }
    return l_;
}

bool LatticeNetwork::loadGrammar(const std::string &file)
{
    delete g_;
    log("loading grammar fst: %s", file.c_str());
    g_ = new GrammarFst();
    return g_->load(file);
}

FstLib::StdFst* LatticeNetwork::getLmLattice(const Lattice &lattice) const
{
    FstLib::StdFst *result = 0;
    if (g_) {
        result = new FstLib::StdComposeFst(
                RmScoreLattice(lattice, LatticeRmScoreMapper()),
                *g_->getFst());
    } else {
        result = new LmScoreLattice(lattice, LatticeLmScoreMapper());
    }
    return result;
}

void LatticeNetwork::createNetwork(const FstLib::StdFst &lmLattice)
{
    /*
    typedef FstLib::LookAheadMatcher<FstLib::StdFst> M;
    typedef FstLib::LookAheadComposeFilter<
            FstLib::AltSequenceComposeFilter<M> > ComposeFilter;
    FstLib::ComposeFstOptions<FstLib::StdArc, M, ComposeFilter> options;
    options.gc_limit = 0;
    */
    if (l_) {
        // OpenFst::VectorFst(FstLib::StdProjectFst(*lmLattice, FstLib::PROJECT_OUTPUT));
        GrammarFst g(OpenFst::VectorFst(FstLib::StdProjectFst(lmLattice, FstLib::PROJECT_OUTPUT)));
        // FstLib::LabelLookAheadRelabeler<FstLib::StdArc>::Relabel(&g, *l_, true);
        l_->relabel(&g);
        AbstractStateTable *states = 0;
        AbstractLexicalFst::ComposeFst *composed = l_->compose(g, 0, &states);
        *f_ = *composed;
        delete composed;
        // *f_ = FstLib::StdComposeFst(*l_, g, options);
    } else {
        *f_ = lmLattice;
    }
}

bool LatticeNetwork::loadLattice(const std::string &name)
{
    const Lattice *lattice = archive_->read(name, false);
    if (!lattice) {
        error("cannot load lattice '%s'", name.c_str());
        return false;
    }
    bool flagSymbols = FLAGS_fst_compat_symbols;
    FLAGS_fst_compat_symbols = false;
    FstLib::Fst<OpenFst::Arc> *lmLattice = getLmLattice(*lattice);
    createNetwork(*lmLattice);
    delete lmLattice;
    delete lattice;
    if (g_ || l_) FstLib::Connect(f_);
    FstLib::ArcSort(f_, FstLib::StdILabelCompare());
    FLAGS_fst_compat_symbols = flagSymbols;
    return true;
}
