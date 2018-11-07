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
#include "tDraw.hh"
#include "tOutput.hh"
#include "Resources.hh"
#include "Output.hh"

namespace Fsa {
    /**
     * storage
     **/
    bool write(      ConstAutomatonRef f, const std::string &format, std::ostream &o, StoredComponents what, bool progress)
    { return Ftl::write<Automaton>(getResources(), f, format, o, what, progress); }

    bool write(      ConstAutomatonRef f, const std::string &file, StoredComponents what, bool progress)
    { return Ftl::write<Automaton>(getResources(), f, file, what, progress); }

    bool writeAtt(   ConstAutomatonRef f, std::ostream &o, StoredComponents what, bool progress)
    { return Ftl::writeAtt<Automaton>(getResources(), f, o, what, progress); }

    bool writeAtt(   ConstAutomatonRef f, const std::string &file, StoredComponents what, bool progress)
    { return Ftl::writeAtt<Automaton>(getResources(), f, file, what, progress); }

    bool writeBinary(ConstAutomatonRef f, std::ostream &o, StoredComponents what, bool progress)
    { return Ftl::writeBinary<Automaton>(getResources(), f, o, what, progress); }

    bool writeBinary(ConstAutomatonRef f, const std::string &file, StoredComponents what, bool progress)
    { return Ftl::writeBinary<Automaton>(getResources(), f, file, what, progress); }

    bool writeLinear(ConstAutomatonRef f, std::ostream &o, StoredComponents what, bool progress, bool printAll)
    { return Ftl::writeLinear<Automaton>(getResources(), f, o, what, progress, printAll); }

    bool writeLinear(ConstAutomatonRef f, const std::string &file, StoredComponents what, bool progress)
    { return Ftl::writeLinear<Automaton>(getResources(), f, file, what, progress); }

    bool writeXml(   ConstAutomatonRef f, std::ostream &o, StoredComponents what, bool progress)
    { return Ftl::writeXml<Automaton>(getResources(), f, o, what, progress); }

    bool writeXml(   ConstAutomatonRef f, const std::string &file, StoredComponents what, bool progress)
    { return Ftl::writeXml<Automaton>(getResources(), f, file, what, progress); }

    bool writeTrXml(  ConstAutomatonRef f, std::ostream &o, StoredComponents what, bool progress)
    { return Ftl::writeTrXml<Automaton>(getResources(), f, o, what, progress); }

    bool writeTrXml(  ConstAutomatonRef f, const std::string &file, StoredComponents what, bool progress)
    { return Ftl::writeTrXml<Automaton>(getResources(), f, file, what, progress); }

#if 0
#include <unistd.h>
#include <sys/mman.h>

    class Mmapped : public Automaton {
    private:
        int fd_;
        void *file_;
        size_t length_;
        off_t inputOffset_, outputOffset_;
    public:
        Mmapped(const std::string &file) {
            char magic[8];
            bi.read(magic, 8);
            if (strcmp(magic, "RWTHFSA") != 0) return false;

            u32 tmp;
            bi >> tmp;
            setType(Type(tmp));
            bi >> tmp;
            tmp |= PropertyStorage;
            setProperties(tmp, tmp);
            bi >> tmp;
            setSemiring(getSemiring(SemiringType(tmp)));

            StaticAlphabet *a = new StaticAlphabet();
            if (!a->read(bi)) return false;
            if (type() == TypeTransducer) {
                a = new StaticAlphabet();
                if (!a->read(bi)) return false;
            }

            // mmap rest of file
            fd_ = open(path);
            if (fd_ >= 0) {
                length_ = ;
                file_ = mmap(0, length_, PROT_READ, MAP_SHARED, fd_, 0);
            } else std::cerr << "could not open file '" << file << "' as fsa (mmapped)." << std::endl;
        }
        virtual ~Mmapped() {
            if (fd_) {
                munmap(file_, length_);
                close(fd_);
            }
        }

        virtual ConstAlphabetRef getInputAlphabet() const { return MmappedAlphabet(fd_, inputOffset_); }
        virtual ConstAlphabetRef getOutputAlphabet() const { return MmappedAlphabet(fd_, outputOffset_); }
        virtual ConstStateRef getState(StateId s) {
            if (s < nStates_) return states_[id];
            return 0;
        }
        virtual void releaseState(StateId s) {}
    };
#endif

#if 0
    bool writeHtk(   ConstAutomatonRef f, std::ostream &o, StoredComponents what, bool progress) {}
    bool writeHtk(   ConstAutomatonRef f, const std::string &file, StoredComponents what, bool progress) {}
#endif


    /**
     * visualization
     **/
    bool drawDot(ConstAutomatonRef f, std::ostream &o, Fsa::Hint hint, bool progress)
    { return Ftl::drawDot<Automaton>(f, o, hint, progress); }
    bool drawDot(ConstAutomatonRef f, const std::string &file, Fsa::Hint hint, bool progress)
    { return Ftl::drawDot<Automaton>(f, file, hint, progress); }
} // namespace Fsa
