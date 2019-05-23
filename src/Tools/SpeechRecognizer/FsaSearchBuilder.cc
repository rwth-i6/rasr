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
#include <Am/Module.hh>
#include <Core/Application.hh>
#include <OpenFst/Module.hh>
#include <Search/Wfst/Builder.hh>
#include <Search/Wfst/Module.hh>
#include <Speech/ModelCombination.hh>
#include <sstream>

using namespace Search::Wfst;

/**
 * applies a sequence of operations to automata.
 * the automata produced / modified are organized in
 * a stack.
 */
class FsaSearchBuilderTool : public virtual Core::Application {
public:
    FsaSearchBuilderTool()
            : Core::Application() {
        INIT_MODULE(Am);
        INIT_MODULE(OpenFst);
        INIT_MODULE(Search::Wfst);
        setTitle("fsa-search-builder");
        setDefaultLoadConfigurationFile(false);
        setDefaultOutputXmlHeader(false);
    }

private:
    static const Core::ParameterStringVector paramOperations;
    typedef std::vector<Builder::Operation*> OperationList;

    std::pair<std::string, std::string> getOperationAndName(const std::string& spec) const {
        std::string op, name;
        op = name                  = spec;
        std::string::size_type pos = spec.find(",");
        if (pos != std::string::npos) {
            op   = spec.substr(0, pos);
            name = spec.substr(pos + 1);
        }
        log() << "operation '" << op << "' with name '" << name << "'";
        return std::make_pair(op, name);
    }

    void getOperationList(Builder::Resources& r, OperationList& l) {
        l.clear();
        std::vector<std::string> operationNames = paramOperations(config);

        const Search::Wfst::Module_& module = Search::Wfst::Module::instance();
        for (std::vector<std::string>::const_iterator i = operationNames.begin(); i != operationNames.end(); ++i) {
            std::string op, name;
            Core::tie(op, name)   = getOperationAndName(*i);
            Builder::Operation* o = module.getBuilderOperation(op, select(name), r);
            if (!o) {
                error("unknown operation '%s'", op.c_str());
            }
            else {
                l.push_back(o);
            }
        }
        log("%d operations", int(l.size()));
    }

    bool runOperations(OperationList& ops) {
        typedef std::vector<Builder::Operation::AutomatonRef> AutomatonStack;
        AutomatonStack                                        stack;
        for (OperationList::iterator iOp = ops.begin(); iOp != ops.end(); ++iOp) {
            Builder::Operation& op    = **iOp;
            u32                 input = op.nInputAutomata();

            if (stack.size() < input) {
                error("operation %s requires %d operands, but stack size is: %d", op.name().c_str(), input, int(stack.size()));
                return false;
            }
            for (u32 i = 0; i < input; ++i) {
                if (!op.addInput(stack[stack.size() - (i + 1)])) {
                    error("cannot set input for operation '%s'", op.name().c_str());
                    return false;
                }
            }
            if (op.consumeInput()) {
                for (u32 i = 0; i < input; ++i) {
                    stack.pop_back();
                }
            }
            Builder::Operation::AutomatonRef result = op.getResult();
            if (!result) {
                if (op.hasOutput()) {
                    error("operation '%s' could not produce any output", op.name().c_str());
                    return false;
                }
            }
            else {
                stack.push_back(result);
            }
        }
        log("%d automata remaining", int(stack.size()));
        while (!stack.empty()) {
            delete stack.back();
            stack.pop_back();
        }
        return true;
    }

public:
    std::string getParameterDescription() const {
        std::ostringstream out;
        paramOperations.printShortHelp(out);
        std::vector<std::string> opNames = Search::Wfst::Module::instance().builderOperations();
        out << "available operations:\n    ";
        std::copy(opNames.begin(), opNames.end(), std::ostream_iterator<std::string>(out, "\n    "));
        return out.str();
    }
    int main(const std::vector<std::string>& arguments) {
        Builder::Resources resources(config);
        OperationList      ops;
        getOperationList(resources, ops);
        if (ops.empty()) {
            return 0;
        }
        if (!runOperations(ops)) {
            error("not all operations have been applied");
            return 1;
        }
        for (OperationList::const_iterator o = ops.begin(); o != ops.end(); ++o)
            delete *o;
        return 0;
    }
};

const Core::ParameterStringVector FsaSearchBuilderTool::paramOperations(
        "operations", "operations applied to a stack of automata. list separated by ' ', format <operation>,<name>", " ");

APPLICATION(FsaSearchBuilderTool)
