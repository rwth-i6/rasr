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
#include <Search/Wfst/IoOperations.hh>
#include <Search/Wfst/CompressedNetwork.hh>
#include <Search/Wfst/Network.hh>
#include <Fsa/Input.hh>
#include <Fsa/Output.hh>
#include <OpenFst/Output.hh>
#include <OpenFst/Input.hh>
#include <fst/const-fst.h>
#include <fst/compact-fst.h>

using namespace Search::Wfst::Builder;

const Core::ParameterString FileOperation::paramFilename(
    "filename", "filename", "");
const Core::Choice FileOperation::choiceType(
    "vector", TypeVector,
    "const", TypeConst,
    "compact", TypeCompact,
    Core::Choice::endMark());
const Core::ParameterChoice FileOperation::paramType(
    "type", &choiceType, "fst type", TypeVector);

std::string FileOperation::filename() const
{
    return paramFilename(config);
}

bool FileOperation::precondition() const
{
    if (filename().empty()) {
        error("no filename given");
        return false;
    }
    return true;
}

const Core::ParameterStringVector ReadOperation::paramAttributes(
        "attributes",
        "attributes to attach to the transducer read, format: key:value,key:value",
        ",");

void ReadOperation::attachAttributes(AutomatonRef automaton) const
{
    std::vector<std::string> attributes, buffer;
    attributes = paramAttributes(config);
    for (std::vector<std::string>::const_iterator a = attributes.begin();
            a != attributes.end(); ++a) {
        buffer = Core::split(*a, ":");
        verify(buffer.size() == 2);
        automaton->setAttribute(buffer[0], buffer[1]);
        log("attribute '%s' = '%s'", buffer[0].c_str(), buffer[1].c_str());
    }
}

Operation::AutomatonRef ReadFst::process()
{
    AutomatonRef r = new Automaton;
    switch (static_cast<FileType>(paramType(config))) {
    case TypeVector: {
        OpenFst::VectorFst *i = OpenFst::VectorFst::Read(filename());
        if (!i) {
            error("cannot read %s", filename().c_str());
        }
        FstLib::Cast(*i, r);
        delete i;
        break;
    }
    case TypeConst: {
        FstLib::StdConstFst *i = FstLib::StdConstFst::Read(filename());
        if (!i) {
            error("cannot read %s", filename().c_str());
        }
        FstLib::Cast(*i, r);
        delete i;
        break;
    }
    default:
        defect();
    }
    log("read %s", filename().c_str());
    attachAttributes(r);
    return r;
}

Operation::AutomatonRef ReadFsa::process()
{
    Fsa::ConstAutomatonRef fsa = Fsa::read(filename());
    if (!fsa) {
        error("cannot read %s", filename().c_str());
    }
    log("read %s", filename().c_str());
    AutomatonRef a = OpenFst::convertFromFsa<Fsa::Automaton, Automaton>(fsa);
    attachAttributes(a);
    return a;
}

bool WriteOperation::precondition() const
{
    return SleeveOperation::precondition() && FileOperation::precondition();
}

Operation::AutomatonRef WriteFst::process()
{
    bool writeOk = false;
    switch (static_cast<FileType>(paramType(config))) {
    case TypeVector:
        writeOk = input_->Write(filename());
        break;
    case TypeConst:
        writeOk = convertAndWrite<FstLib::StdConstFst>(filename());
        break;
    case TypeCompact:
        writeOk = convertAndWrite<FstLib::StdCompactAcceptorFst>(filename());
        break;
    default:
        defect();
    }
    if(!writeOk) {
        FileOperation::error("cannot write %s", filename().c_str());
    } else {
        log("wrote %s", filename().c_str());
    }
    return input_;
}

template<class F>
bool WriteFst::convertAndWrite(const std::string &filename) const
{
    typedef F TargetType;
    F convert(*input_);
    return convert.Write(filename);
}

Operation::AutomatonRef WriteFsa::process()
{
    Fsa::ConstAutomatonRef fsa = OpenFst::convertToFsa(*input_, Fsa::TropicalSemiring);
    if (!Fsa::write(fsa, filename())) {
        FileOperation::error("cannot write %s", filename().c_str());
    } else {
        log("wrote %s", filename().c_str());
    }
    return input_;
}

Operation::AutomatonRef Compress::process()
{
    CompressedNetwork network(config, false);
    if (!network.build(input_, false)) {
        FileOperation::error("cannot build compressed network");
    } else {
        if (!network.write(filename())) {
            FileOperation::error("cannot write %s", filename().c_str());
        } else {
            log("wrote %s", filename().c_str());
        }
    }
    return input_;
}
