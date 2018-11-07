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
#include <Modules.hh>
#include <Core/Application.hh>
#include <Core/Debug.hh>
#include <Search/Wfst/Network.hh>
#include <Search/Wfst/DynamicLmFst.hh>
#include <fst/compose.h>
#include <fst/arcsort.h>
#include <fst/determinize.h>
#include <Search/Wfst/Lattice.hh>
#include <OpenFst/Types.hh>
#include <OpenFst/Count.hh>

class TestApplication : public virtual Core::Application
{
public:
  /**
   * Standard constructor for setting title.
   */
  TestApplication ( ) : Core::Application ( ) {
      setTitle ( "check" );
  }

  std::string getUsage() const { return "test network\n"; }


  int main(const std::vector<std::string> &arguments)
  {
      log("reading ") << arguments[0];
      Search::Wfst::Lattice *l = Search::Wfst::Lattice::Read(arguments[0]), det;
      Fsa::AutomatonCounts c = OpenFst::count(*l);
      log("states: %d, arcs: %d", c.nStates_, c.nArcs_);
      FstLib::Determinize(*l, &det);
      log("writing ") << arguments[1];
      det.Write(arguments[1]);
      return 0;

#if 0
      Bliss::LexiconRef lexicon = Bliss::Lexicon::create(select("lexicon"));
      Core::Ref<Lm::LanguageModel> lm = Lm::Module::instance().createLanguageModel(select("lm"), lexicon);
      Search::DynamicLmFstOptions opts;
      opts.lm = lm;
      Search::DynamicLmFst lmFst(opts);
      std::cout << "Start: " << lmFst.Start() << std::endl;
      std::cout << "Start: " << lmFst.Start() << std::endl;
      const OpenFst::SymbolTable *symbols = lmFst.InputSymbols();
      OpenFst::VectorFst words;
      words.SetStart(words.AddState());
      OpenFst::StateId s1 = words.AddState();
      OpenFst::StateId sEnd = words.AddState();
      words.SetFinal(sEnd, OpenFst::Weight::One());
      OpenFst::Label lAnd = symbols->Find("AND"), lSixty = symbols->Find("SIXTY");
      verify(lAnd > 0);
      verify(lSixty > 0);
      words.AddArc(words.Start(), OpenFst::Arc(lAnd, lAnd, OpenFst::Weight::One(), s1));
      words.AddArc(s1, OpenFst::Arc(lSixty, lSixty, OpenFst::Weight::One(), sEnd));
      words.SetOutputSymbols(symbols);
      FstLib::ArcSort(&words, FstLib::ILabelCompare<OpenFst::Arc>());
      std::cout << "createing compose fst" << std::endl;
      FstLib::ComposeFstOptions<OpenFst::Arc> composeOpt;
      FstLib::ComposeFst<OpenFst::Arc> *compose = new FstLib::ComposeFst<OpenFst::Arc>(words, lmFst, composeOpt);
      OpenFst::VectorFst result;
      std::cout << "traversing compose fst" << std::endl;
      result = *compose;
      result.Write("/tmp/result.fst");
      delete compose;
#endif
#if 0
      s32 ival = 0;
      std::string sval = "4096";
      Core::strconv(sval, ival);
      std::cout << ival << std::endl;
      return 0;
      Core::Configuration recognizerConfig(select("speech-recognizer"), "recognizer");
      Search::ComposedNetwork network(Core::Configuration(recognizerConfig, "network"));
      network.reset();
      Search::ComposedNetwork::StateIndex s = network.initialStateIndex();
      Search::ComposedNetwork::EpsilonArcIterator iter(&network, s);
      while (!iter.done()) {
          const Search::ComposedNetwork::EpsilonArc &arc = iter.value();
          std::cout << arc.olabel << " " << arc.nextstate << " " << arc.weight << std::endl;
          iter.next();
      }
#endif
      return EXIT_SUCCESS;
  } //end main

protected:
};

APPLICATION(TestApplication)
