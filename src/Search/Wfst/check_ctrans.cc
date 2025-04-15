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
#include <Am/ClassicStateModel.hh>
#include <Am/Module.hh>
#include <Bliss/Phoneme.hh>
#include <Core/Application.hh>
#include <Core/Debug.hh>
#ifndef CMAKE_DISABLE_MODULES_HH
#include <Modules.hh>
#endif
#include <OpenFst/SymbolTable.hh>
#include <OpenFst/Types.hh>
#include <fst/arcsort.h>
#include <fst/compose.h>
#include <fst/determinize.h>

/**
 * Test suite for the Search module of SPRINT.
 *
 */
class TestApplication : public virtual Core::Application {
public:
    /**
     * Standard constructor for setting title.
     */
    TestApplication()
            : Core::Application() {
        INIT_MODULE(Am);
        setTitle("checkc");
    }

    std::string getUsage() const {
        return "test c transducer\n";
    }

    void createPhoneSequence(const Am::Allophone& a, OpenFst::VectorFst* seq) const {
        Bliss::Phoneme::Id history = (a.history().size() ? a.history()[0] : Bliss::Phoneme::term);
        Bliss::Phoneme::Id central = a.central();
        Bliss::Phoneme::Id future  = (a.future().size() ? a.future()[0] : Bliss::Phoneme::term);
        OpenFst::Label     labels[4];
        labels[0] = history == Bliss::Phoneme::term ? sil_ : history;
        labels[1] = central;
        labels[2] = future == Bliss::Phoneme::term ? sil_ : future;
        labels[3] = seqEnd_;
        for (int i = 0; i < 4; ++i)
            labels[i] = labels[i];
        labels[0] += initialOffset_;

        if (a.boundary & Am::Allophone::isInitialPhone)
            labels[1] += initialOffset_;
        if (a.boundary & Am::Allophone::isFinalPhone)
            labels[2] += initialOffset_;

        if (verbose_) {
            for (int i = 0; i < 4; ++i)
                std::cout << labels[i] << "=" << phoneSymbols_->Find(labels[i]) << " ";
            std::cout << std::endl;
        }
        seq->DeleteStates();
        OpenFst::StateId s = seq->AddState(), ns;
        seq->SetStart(s);
        for (int i = 0; i < 4; ++i) {
            ns = seq->AddState();
            seq->AddArc(s, OpenFst::Arc(labels[i], labels[i], OpenFst::Weight::One(), ns));
            s = ns;
        }
        seq->SetFinal(s, OpenFst::Weight::One());
    }

    void createPhoneSequence(const Bliss::Phoneme::Id* pron, OpenFst::VectorFst* seq) const {
        seq->DeleteStates();
        OpenFst::StateId s = seq->AddState(), ns = OpenFst::InvalidStateId;
        seq->SetStart(s);
        OpenFst::Label offset = initialOffset_;
        while (*pron != Bliss::Phoneme::term) {
            ns               = seq->AddState();
            OpenFst::Label l = *pron + offset;
            seq->AddArc(s, OpenFst::Arc(l, l, OpenFst::Weight::One(), ns));
            if (verbose_) {
                std::cout << phoneSymbols_->Find(l) << " ";
            }
            s = ns;
            ++pron;
            offset = 0;
        }
        ns                      = seq->AddState();
        OpenFst::Label endPhone = seqEnd_;
        if (verbose_) {
            std::cout << phoneSymbols_->Find(endPhone) << std::endl;
        }
        seq->AddArc(s, OpenFst::Arc(endPhone, endPhone, OpenFst::Weight::One(), ns));
        seq->SetFinal(ns, OpenFst::Weight::One());
    }

    int checkAllophones(Core::Ref<Am::AcousticModel> am, const OpenFst::VectorFst* f) const {
        bool                                        error         = false;
        int                                         nChecked      = 0;
        Core::Ref<const Am::AllophoneAlphabet>      allophones    = am->allophoneAlphabet();
        const Am::AllophoneAlphabet::AllophoneList& allophoneList = allophones->allophones();
        for (Am::AllophoneAlphabet::AllophoneList::const_iterator ai = allophoneList.begin();
             ai != allophoneList.end() && !error; ++ai) {
            const Am::Allophone* a = *ai;
            if (verbose_)
                std::cout << "allophone " << allophones->symbol(allophones->index(a)) << std::endl;
            OpenFst::VectorFst phoneSeq, result;
            phoneSeq.SetProperties(FstLib::kAcceptor | FstLib::kNotAcceptor, FstLib::kAcceptor | ~FstLib::kNotAcceptor);
            createPhoneSequence(*a, &phoneSeq);
            FLAGS_fst_compat_symbols = false;
            FstLib::Compose(*f, phoneSeq, &result);
            if (verbose_)
                printAllophoneSequence(allophones, result);
            if (result.NumStates() < 5) {
                error = true;
                std::cout << "ERROR: wrong number of states: " << result.NumStates() << std::endl;
            }
            if (!isLinearFst(result)) {
                error = true;
                std::cout << "ERROR: result not linear" << std::endl;
            }
            OpenFst::StateId s        = result.Start();
            u32              inputPos = 0;
            while (s != OpenFst::InvalidStateId && inputPos < 2) {
                OpenFst::ArcIterator aiter(result, s);
                if (aiter.Done()) {
                    std::cout << "ERROR: result too short" << std::endl;
                    error = true;
                    break;
                }
                const OpenFst::Arc& arc = aiter.Value();
                if (verbose_) {
                    std::cout << inputPos << " " << arc.ilabel << " " << allophones->symbol(OpenFst::convertLabelToFsa(arc.ilabel)) << std::endl;
                }
                if (arc.ilabel != OpenFst::Epsilon) {
                    ++inputPos;
                    if (inputPos == 2 && OpenFst::convertLabelToFsa(arc.ilabel) != allophones->index(*a)) {
                        error = true;
                        std::cout << "ERROR: wrong input label: " << arc.ilabel << " "
                                  << allophones->symbol(OpenFst::convertLabelToFsa(arc.ilabel)) << std::endl;
                        break;
                    }
                }
                s = arc.nextstate;
            }
            if (error) {
                Fsa::LabelId id = allophones->index(*a);
                std::cout << "allophone: " << id << " "
                          << allophones->symbol(id) << std::endl;
                result.SetOutputSymbols(phoneSymbols_);
                result.Write(Core::form("/tmp/%d_result.fst", id));
                phoneSeq.SetInputSymbols(phoneSymbols_);
                phoneSeq.Write(Core::form("/tmp/%d_seq.fst", id));
            }
            ++nChecked;
        }
        return nChecked;
    }

    bool isLinearFst(const OpenFst::VectorFst& result) const {
        for (OpenFst::StateIterator si(result); !si.Done(); si.Next()) {
            if (result.NumArcs(si.Value()) > 1) {
                return false;
            }
        }
        return true;
    }

    void printAllophoneSequence(Core::Ref<const Am::AllophoneAlphabet> allophones,
                                const OpenFst::VectorFst&              result) const {
        OpenFst::StateId s = result.Start();
        while (s != OpenFst::InvalidStateId) {
            OpenFst::ArcIterator aiter(result, s);
            if (!aiter.Done()) {
                std::cout << "  " << aiter.Value().ilabel << " " << allophones->symbol(OpenFst::convertLabelToFsa(aiter.Value().ilabel)) << std::endl;
            }
            else {
                break;
            }
            s = aiter.Value().nextstate;
        }
    }

    bool checkWord(Core::Ref<const Am::AllophoneAlphabet> allophones, Core::Ref<const Bliss::PhonemeInventory> pi,
                   const Bliss::Phoneme::Id* pron, const OpenFst::VectorFst& result) const {
        OpenFst::StateId   s          = result.Start();
        Bliss::Phoneme::Id prev       = Bliss::Phoneme::term;
        u32                phoneIndex = 0;
        while (*pron != Bliss::Phoneme::term) {
            OpenFst::ArcIterator aiter(result, s);
            if (aiter.Done()) {
                std::cout << "ERROR: result too short" << std::endl;
                return false;
            }
            const OpenFst::Arc& arc = aiter.Value();
            if (arc.ilabel != OpenFst::Epsilon) {
                const Am::Allophone* allophone = allophones->allophone(OpenFst::convertLabelToFsa(arc.ilabel));
                verify(allophone);
                Bliss::Phoneme::Id phoneseq[3];
                phoneseq[0] = (allophone->history().size() ? allophone->history()[0] : Bliss::Phoneme::term);
                phoneseq[1] = allophone->central();
                phoneseq[2] = (allophone->future().size() ? allophone->future()[0] : Bliss::Phoneme::term);
                Bliss::Phoneme::Id ref[3];
                for (u32 i = 0; i < 3; ++i) {
                    ref[i]               = (i ? *(pron + i - 1) : prev);
                    const bool isCiPhone = (ref[i] != Bliss::Phoneme::term && !pi->phoneme(ref[i])->isContextDependent());
                    if (isCiPhone) {
                        if (i == 1) {
                            ref[0] = ref[2] = Bliss::Phoneme::term;
                            break;
                        }
                        else {
                            ref[i] = Bliss::Phoneme::term;
                        }
                    }
                }
                for (u32 i = 0; i < 3; ++i) {
                    if (ref[i] != phoneseq[i]) {
                        std::cout << "ERROR: wrong phone pos=" << phoneIndex << " c=" << i << " "
                                  << "expected " << static_cast<u32>(ref[i]) << " "
                                  << (ref[i] == Bliss::Phoneme::term ? "#" : pi->phoneme(ref[i])->symbol().str()) << " "
                                  << "found " << static_cast<u32>(phoneseq[i]) << " "
                                  << (phoneseq[i] == Bliss::Phoneme::term ? "#" : pi->phoneme(phoneseq[i])->symbol().str())
                                  << std::endl;
                        return false;
                    }
                }
                prev = *pron;
                ++pron;
                ++phoneIndex;
            }
            s = arc.nextstate;
        }
        return true;
    }

    int checkWords(Core::Ref<Am::AcousticModel> am, Bliss::LexiconRef lexicon, const OpenFst::VectorFst* f) const {
        int                                      nChecked = 0;
        Bliss::Lexicon::PronunciationIterator    pronIter, pronEnd;
        Core::Ref<const Bliss::PhonemeInventory> pi       = lexicon->phonemeInventory();
        Core::tie(pronIter, pronEnd)                      = lexicon->pronunciations();
        Core::Ref<const Am::AllophoneAlphabet> allophones = am->allophoneAlphabet();
        OpenFst::VectorFst                     phoneSeq, result;
        bool                                   error = false;
        for (; pronIter != pronEnd; ++pronIter) {
            const Bliss::Pronunciation* pron = *pronIter;
            if (verbose_) {
                std::cout << "pronunciation: ";
                std::cout << pron->format(pi) << std::endl;
            }
            const Bliss::Phoneme::Id* phones = pron->phonemes();
            createPhoneSequence(phones, &phoneSeq);
            phoneSeq.SetInputSymbols(phoneSymbols_);
            FLAGS_fst_compat_symbols = false;
            FstLib::Compose(*f, phoneSeq, &result);
            if (verbose_ && result.NumStates()) {
                printAllophoneSequence(allophones, result);
            }
            if (!isLinearFst(result)) {
                std::cout << "ERROR: result not linear" << std::endl;
                error = true;
            }
            if (result.NumStates() != phoneSeq.NumStates()) {
                std::cout << "ERROR: wrong number of states. "
                          << "expected: " << phoneSeq.NumStates() << " "
                          << "found: " << result.NumStates() << std::endl;
                error = true;
            }
            if (!error) {
                if (!checkWord(allophones, pi, phones, result)) {
                    std::cout << "ERROR: wrong allophone sequence " << std::endl;
                    error = true;
                }
            }
            if (error) {
                break;
            }
            ++nChecked;
        }
        return nChecked;
    }

    int main(const std::vector<std::string>& arguments) {
        phoneSymbols_ = 0;
        verbose_      = paramVerbose(config);
        std::cout << "reading " << arguments[0] << std::endl;
        OpenFst::VectorFst*          f       = OpenFst::VectorFst::Read(arguments[0]);
        Bliss::LexiconRef            lexicon = Bliss::Lexicon::create(select("lexicon"));
        Core::Ref<Am::AcousticModel> am      = Am::Module::instance().createAcousticModel(
                select("acoustic-model"), lexicon, Am::AcousticModel::noEmissions);
        Core::Ref<const Bliss::PhonemeInventory> pi = lexicon->phonemeInventory();
        sil_                                        = pi->phoneme(paramSilencePhone(config))->id();
        initialOffset_                              = paramInitialPhoneOffset(config);
        phoneSymbols_                               = f->OutputSymbols();
        verify(phoneSymbols_);
        log("sequence end symbol: %s", paramSequenceEnd(config).c_str());
        seqEnd_ = phoneSymbols_->Find(paramSequenceEnd(config));
        verify(seqEnd_ > 0);
        u32 nChecked = checkAllophones(am, f);
        log() << nChecked << " allophones checked";
        nChecked = checkWords(am, lexicon, f);
        log() << nChecked << " words checked";
        delete f;
        return EXIT_SUCCESS;
    }  // end main

protected:
    static const Core::ParameterInt    paramInitialPhoneOffset;
    static const Core::ParameterString paramSilencePhone;
    static const Core::ParameterBool   paramVerbose;
    static const Core::ParameterString paramSequenceEnd;
    Bliss::Phoneme::Id                 sil_;
    OpenFst::Label                     initialOffset_;
    OpenFst::Label                     seqEnd_;
    Core::Ref<Am::AcousticModel>       am;
    const OpenFst::SymbolTable*        phoneSymbols_;
    bool                               verbose_;
};

const Core::ParameterInt TestApplication::paramInitialPhoneOffset(
        "initial-phone-offset", "initial phone offset", 0);

const Core::ParameterString TestApplication::paramSilencePhone(
        "silence-phone", "silence phoneme", "si");

const Core::ParameterString TestApplication::paramSequenceEnd(
        "sequence-end", "sequence end symbol", "si@i");

const Core::ParameterBool TestApplication::paramVerbose(
        "verbose", "verbose output", false);
APPLICATION(TestApplication)
