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
#include <Core/Application.hh>
#include <Core/CompressedStream.hh>
#include <Core/Hash.hh>
#include <Core/Statistics.hh>
#include <Core/Tokenizer.hh>

#include <Fsa/Arithmetic.hh>
#include <Fsa/Automaton.hh>
#include <Fsa/Basic.hh>
#include <Fsa/Best.hh>
#include <Fsa/Cache.hh>
#include <Fsa/Compose.hh>
#include <Fsa/Determinize.hh>
#include <Fsa/Input.hh>
#include <Fsa/Levenshtein.hh>
#include <Fsa/Minimize.hh>
#include <Fsa/Output.hh>
#include <Fsa/Packed.hh>
#include <Fsa/Permute.hh>
#include <Fsa/Project.hh>
#include <Fsa/Prune.hh>
#include <Fsa/Random.hh>
#include <Fsa/Rational.hh>
#include <Fsa/RealSemiring.hh>
#include <Fsa/RemoveEpsilons.hh>
#include <Fsa/Sort.hh>
#include <Fsa/Sssp.hh>
#include <Fsa/Sssp4SpecialSymbols.hh>
#include <Fsa/Stack.hh>
#include <Fsa/Static.hh>
#include <Fsa/Types.hh>

#ifndef CMAKE_DISABLE_MODULE_HH
#include <Modules.hh>
#endif
#ifdef MODULE_OPENFST
#include <OpenFst/Module.hh>
#endif

using namespace Fsa;

class FsaTool : public Core::Application {
private:
    enum Operation {
        op_best,
        op_cache,
        op_closure,
        op_collect,
        op_complement,
        op_compose,
        op_concat,
        op_copy,
        op_count,
        op_default,
        op_delete,
        op_determinize,
        op_difference,
        op_draw,
        op_duplicate,
        op_expm,
        op_extend,
        op_fuse,
        op_info,
        op_invert,
        op_levenshtein,
        op_map_input,
        op_map_output,
        op_memory,
        op_minimize,
        op_multiply,
        op_normalize,
        op_partial,
        op_permute,
        op_posterior,
        op_posterior_64,
        op_posterior_expectation,
        op_posterior_fail,
        op_project,
        op_prune,
        op_push,
        op_random,
        op_remove,
        op_semiring,
        op_sort,
        op_sync_prune,
        op_time,
        op_transpose,
        op_trim,
        op_unite,
        op_wait,
        op_write
    };
    Operation                  operation_;
    static Core::Choice        OperationChoice;
    static Core::ParameterBool paramProgress;

    Stack<ConstAutomatonRef> stack_;
    ConstAutomatonRef        first_, second_;
    ConstSemiringRef         semiring_;

private:
    void oneOperand() {
        if (stack_.size() < 1) {
            std::cerr << OperationChoice[operation_] << ": needs one operand from stack" << std::endl;
            ::exit(0);
        }
        else
            first_ = stack_.pop();
    }
    void twoOperands() {
        if (stack_.size() < 2) {
            std::cerr << OperationChoice[operation_] << ": needs two operands from stack" << std::endl;
            ::exit(0);
        }
        else {
            second_ = stack_.pop();
            first_  = stack_.pop();
        }
    };

public:
    FsaTool()
            : semiring_(TropicalSemiring) {
#ifdef MODULE_OPENFST
        INIT_MODULE(OpenFst)
#endif
        setTitle("fsa");
        setDefaultLoadConfigurationFile(false);
        setDefaultOutputXmlHeader(false);
        config.set("fsa.log.channel", "stderr");
        config.set("fsa.warning.channel", "stderr");
        config.set("fsa.error.channel", "stderr");
        config.set("fsa.critical.channel", "stderr");
    }
    std::string getUsage() const {
        std::string usage = "\n"
                            "fsa [OPTION(S)] <FILE | OPERATION> ...\n"
                            "\n"
                            "options:\n"
                            "   --help            print this page\n"
                            "   --progress=yes    show progress during operations\n"
                            "   --resources=yes   print resource database\n"
                            "\n"
                            "algorithms (parameters and defaults in brackets, use e.g. closure,kleene or nbest,n=100):\n"
                            "   best          extract [n(1)] best path(s)\n"
                            "   closure       [kleene] closure of the topmost automaton\n"
                            "   collect       collect each arc weight and [value]\n"
                            "   concat        concat the [n(2)] topmost automata\n"
                            "   complement    automaton that represents the complement language\n"
                            "   compose       compose the two topmost automata [filter=(match),seq]\n"
                            "   determinize   determinize topmost automaton [disambiguate]\n"
                            "   difference    build the difference of the topmost and second topmost automaton\n"
                            "   duplicate     duplicate topmost automaton\n"
                            "   extend        extend each arc weight by [value]\n"
                            "   expm          weight --> exp(-weight)\n"
                            "   invert        swap input and output labels\n"
                            "   levenshtein   calculates the levenshtein distance of the two topmost automata\n"
                            "   map-input     map input labels using output alphabet of second topmost automaton\n"
                            "   map-output    map output labels using input alphabet of second topmost automaton\n"
                            "   multiply      multiply each arc weight by [value] (log and tropical semiring only)\n"
                            "   minimize      minimize topmost automaton\n"
                            "   normalize     normalizes state ids of topmost automaton (i.e. initial = 0, no gaps)\n"
                            "   partial       partial automaton starting at state [id]\n"
                            "   permute       permute automaton with a window of [n=(infinity)], [type=(ibm),inv,itg,local], \n"
                            "                 [prob=(0.0)] OR [dist=(0.0)] with a maximum distortion of [max=(dist> 0 ? 20 : infinity)]\n"
                            "   posterior     calculate arc posterior weights\n"
                            "   posterior64   calculate arc posterior weights (numerically more stable version for log semiring)\n"
                            "   posteriorE    calculate arc posterior weights with expectation semiring\n"
                            "   posteriorFail calculate arc posterior weights with Fail arcs\n"
                            "   project       project [type=(input),output] labels to input labels\n"
                            "   prune         prune arcs using path posterior weights [beam] threshold\n"
                            "   push          push weights [to=(final),initial] state\n"
                            "   random        select a random path\n"
                            "   remove        remove [type=(epsilons),disambiguators] from topmost automaton\n"
                            "   remove        remove arcs with disambiguation symbols or replace by epsilons\n"
                            "   sort          sort all edges by [(arc),input,output,weight]\n"
                            "   sync-prune    prune states using synchronuous state potentials and [beam] threshold\n"
                            "   transpose     reverse the direction of all arcs\n"
                            "   trim          removes all but the connected and disconnected states\n"
                            "   unite         unite the [n(2)] topmost automata\n"
                            "   fuse          fuse initial states of the [n(2)] topmost automata\n"
                            "\n"
                            "output:\n"
                            "   draw          write topmost automaton to [file=(-)] in dot format [best,detailed]\n"
                            "   write         write topmost automaton or [input,output] alphabet (both input and output is possible, too) or only the states [states] to [file=(-)]\n"
                            "\n"
                            "control:\n"
                            "   cache         caches states of topmost transducer\n"
                            "   copy          creates a static copy of the topmost transducer\n"
                            "   delete        delete topmost transducer\n"
                            "   default       set the default semiring for all following read operations (see list below)\n"
                            "   semiring      change the semiring of the topmost automaton (see list below)\n"
                            "\n"
                            "diagnostics:\n"
                            "   count         [(input),output] arc count statistics for [label] or number of [paths]\n"
                            "   info          print sizes of topmost automaton\n"
                            "   memory        print detailed memory info of topmost automaton\n"
                            "   time          print time consumed by preceeding operation\n"
                            "   wait          wait for pressing <ENTER>\n"
                            "\n"
                            "semirings [tolerance=(1) for log]:\n"
                            "   ";
        std::ostringstream tmp;
        SemiringTypeChoice.printIdentifiers(tmp);
        usage += tmp.str() + "\n";
        usage += "\n"
                 "prepend att:/bin:/lin:/xml/trxml: in order to select file format, packed: for\n"
                 "compressed storage and combine: to combine automata from different files\n"
                 "\n";
        return usage;
    }

    void writeAlphabet(ConstAlphabetRef alphabet, const std::string& fname) {
        if (alphabet) {
            Core::CompressedOutputStream o(fname);
            Core::XmlWriter              xo(o);
            xo << Core::XmlOpen("alphabet") << "\n";
            alphabet->writeXml(xo);
            xo << Core::XmlClose("alphabet") << "\n";
        }
        else {
            std::cerr << "Could not write Alphabet" << std::endl;
        }
    }

    class OperationSpecification : public Core::StringHashMap<std::string> {
    public:
        bool has(const std::string& parameter) const {
            return (find(parameter) != end());
        }
    };

    OperationSpecification parseOperation(const std::string& operation) const {
        OperationSpecification specs;
        Core::StringTokenizer  tokenizer(operation, ",");
        for (Core::StringTokenizer::Iterator iToken = tokenizer.begin(); iToken != tokenizer.end(); ++iToken) {
            std::string            token = *iToken;
            std::string::size_type pos   = token.find('=');
            if (iToken == tokenizer.begin()) {
                specs["op"] = token;
            }
            else if (pos == std::string::npos) {
                specs[token]       = "";
                specs[specs["op"]] = token;
            }
            else {
                specs[token.substr(0, pos)] = token.substr(pos + 1);
            }
        }
        return specs;
    }

    int main(const std::vector<std::string>& arguments) {
        if (arguments.size() == 0)
            std::cerr << getUsage();
        size_t       arg = 0;
        Core::Timer* timer[2];
        timer[0] = timer[1] = 0;
        bool progress       = paramProgress(config);
        while (arg < arguments.size()) {
            const std::string&     argument  = arguments[arg++];
            OperationSpecification specs     = parseOperation(argument);
            Core::Choice::Value    operation = OperationChoice[specs["op"]];
            operation_                       = Operation(operation);
            timer[1]                         = new Core::Timer();
            timer[1]->start();
            if (operation == Core::Choice::IllegalValue) {
                ConstAutomatonRef f = read(argument, semiring_);
                if (f)
                    stack_.push(f);
            }
            else {
                switch (operation) {
                    case op_best:
                        oneOperand();
                        if (specs.has("first")) {
                            stack_.push(firstbest(first_));
                        }
                        else if (!specs.has("n")) {
                            stack_.push(best(first_));
                        }
                        else {
                            char*    error;
                            long int s = strtol(specs["n"].c_str(), &error, 10);
                            if ((*error == '\0') && (s >= 0))
                                stack_.push(nbest(first_, s));
                            else
                                std::cerr << "best: n must be a positive integer" << std::endl;
                        }
                        break;
                    case op_closure:
                        oneOperand();
                        if (specs.has("kleene"))
                            stack_.push(kleeneClosure(first_));
                        else
                            stack_.push(closure(first_));
                        break;
                    case op_collect:
                        if (!specs.has("value"))
                            std::cerr << "collect: needs value" << std::endl;
                        else {
                            oneOperand();
                            stack_.push(collect(first_, first_->semiring()->fromString(specs["value"])));
                        }
                        break;
                    case op_concat: {
                        long int n = 2;
                        if (specs.has("n")) {
                            char* error;
                            n = strtol(specs["n"].c_str(), &error, 10);
                            if ((*error != '\0') || (n < 2)) {
                                std::cerr << "concat: [n] must be larger than 1" << std::endl;
                                break;
                            }
                        }
                        if (stack_.size() >= size_t(n)) {
                            Core::Vector<ConstAutomatonRef> automata;
                            for (u32 i = 0; i < u32(n); ++i)
                                automata.push_back(stack_.pop());
                            stack_.push(concat(automata));
                        }
                        else
                            std::cerr << "concat: not enough automata on stack" << std::endl;
                        break;
                    }
                    case op_complement:
                        oneOperand();
                        stack_.push(complement(first_));
                        break;
                    case op_compose:
                        twoOperands();
                        if ((!specs.has("filter")) || (specs["filter"] == "match"))
                            stack_.push(composeMatching(first_, second_));
                        else if (specs["filter"] == "seq")
                            stack_.push(composeSequencing(first_, second_));
                        break;
                    case op_determinize:
                        oneOperand();
                        stack_.push(
                                determinize(
                                        first_,
                                        specs.has("disambiguate")));
                        break;
                    case op_difference:
                        twoOperands();
                        stack_.push(difference(first_, second_));
                        break;
                    case op_duplicate:
                        oneOperand();
                        stack_.push(first_);
                        stack_.push(first_);
                        break;
                    case op_expm:
                        oneOperand();
                        stack_.push(expm(first_));
                        break;
                    case op_extend:
                        if (!specs.has("value"))
                            std::cerr << "extend: needs value" << std::endl;
                        else {
                            oneOperand();
                            stack_.push(extend(first_, first_->semiring()->fromString(specs["value"])));
                        }
                        break;
                    case op_invert:
                        oneOperand();
                        stack_.push(invert(first_));
                        break;
                    case op_levenshtein: {
                        twoOperands();
                        stack_.push(levenshtein(first_, second_));
                        break;
                    }
                    case op_map_input:
                        twoOperands();
                        stack_.push(first_);
                        stack_.push(mapInput(second_, first_->getOutputAlphabet()));
                        break;
                    case op_map_output:
                        twoOperands();
                        stack_.push(first_);
                        stack_.push(mapOutput(second_, first_->getInputAlphabet()));
                        break;
                    case op_minimize:
                        oneOperand();
                        stack_.push(minimize(first_));
                        break;
                    case op_multiply:
                        if (!specs.has("value"))
                            std::cerr << "multiply: needs value" << std::endl;
                        else {
                            oneOperand();
                            stack_.push(multiply(first_, first_->semiring()->fromString(specs["value"])));
                        }
                        break;
                    case op_normalize:
                        oneOperand();
                        stack_.push(normalize(first_));
                        break;
                    case op_partial: {
                        if (!specs.has("id"))
                            std::cerr << "partial: needs initial state [id]" << std::endl;
                        else {
                            oneOperand();
                            char*    error;
                            long int s = strtol(specs["id"].c_str(), &error, 10);
                            if ((*error == '\0') && (s >= 0))
                                stack_.push(partial(first_, s));
                            else
                                std::cerr << "partial: [id] must be a (positive) state id" << std::endl;
                        }
                        break;
                    }
                    case op_posterior:
                        oneOperand();
                        stack_.push(posterior(first_));
                        break;
                    case op_posterior_64: {
                        Weight tmp;
                        oneOperand();
                        stack_.push(posterior64(first_, tmp));
                        log("totalInv: ") << f32(tmp);
                    } break;
                    case op_posterior_expectation: {
                        Weight tmp;
                        bool   vNormalized = specs["v-norm"].empty() or (specs["v-norm"] == "true");
                        twoOperands();
                        stack_.push(posteriorE(first_, second_, tmp, vNormalized));
                        log("expectation: ") << f32(tmp);
                    } break;
                    case op_posterior_fail: {
                        Weight tmp;
                        oneOperand();
                        stack_.push(posterior4SpecialSymbols(first_, tmp));
                        log("totalInv: ") << f32(tmp);
                    } break;
                    case op_project:
                        oneOperand();
                        if (specs.has("output"))
                            stack_.push(projectOutput(first_));
                        else
                            stack_.push(projectInput(first_));
                        break;
                    case op_prune:
                        if (!specs.has("beam"))
                            std::cerr << "prune: needs [beam] threshold" << std::endl;
                        else {
                            oneOperand();
                            stack_.push(prunePosterior(first_, first_->semiring()->fromString(specs["beam"])));
                        }
                        break;
                    case op_push:
                        oneOperand();
                        if (specs.has("initial"))
                            stack_.push(pushToInitial(first_));
                        else
                            stack_.push(pushToFinal(first_));
                        break;
                    case op_random: {
                        oneOperand();
                        char* error;
                        if (specs.has("seed")) {
                            srand48(strtol(specs["seed"].c_str(), &error, 10));
                            if (*error != '\0')
                                std::cerr << "random: invalid seed value" << std::endl;
                        }
                        else {
                            srand48(time(0));
                        }
                        f64 samplingWeight = 0.0;
                        if (specs.has("weight")) {
                            samplingWeight = strtod(specs["weight"].c_str(), &error);
                            if ((*error != '\0'))
                                std::cerr << "random: invalid importance sampling weight value" << std::endl;
                        }
                        u32 maximumSize = 0;
                        if (specs.has("limit")) {
                            maximumSize = strtol(specs["limit"].c_str(), &error, 10);
                            if (*error != '\0')
                                std::cerr << "random: invalid maximum size value" << std::endl;
                        }
                        stack_.push(random(first_, samplingWeight, maximumSize));
                    } break;
                    case op_remove:
                        oneOperand();
                        if (specs.has("disambiguators"))
                            stack_.push(removeDisambiguationSymbols(first_));
                        else
                            stack_.push(removeEpsilons(first_));
                        break;
                    case op_sort:
                        oneOperand();
                        if (specs.has("weight"))
                            stack_.push(sort(first_, SortTypeByWeight));
                        else if (specs.has("input"))
                            stack_.push(sort(first_, SortTypeByInput));
                        else if (specs.has("output"))
                            stack_.push(sort(first_, SortTypeByOutput));
                        else
                            stack_.push(sort(first_, SortTypeByArc));
                        break;
                    case op_sync_prune:
                        if (!specs.has("beam"))
                            std::cerr << "sync-prune: needs [beam] threshold" << std::endl;
                        else {
                            oneOperand();
                            stack_.push(pruneSync(first_, first_->semiring()->fromString(specs["beam"])));
                        }
                        break;
                    case op_transpose:
                        oneOperand();
                        stack_.push(transpose(first_, progress));
                        first_.reset();
                        break;
                    case op_trim:
                        oneOperand();
                        stack_.push(trim(first_, progress));
                        break;
                    case op_unite: {
                        long int n = 2;
                        if (specs.has("n")) {
                            char* error;
                            n = strtol(specs["n"].c_str(), &error, 10);
                            if ((*error != '\0') || (n < 1)) {
                                std::cerr << "unite: n must be larger than 0" << std::endl;
                                break;
                            }
                        }
                        if (stack_.size() >= size_t(n)) {
                            Core::Vector<ConstAutomatonRef> automata;
                            for (u32 i = 0; i < u32(n); ++i)
                                automata.push_back(stack_.pop());
                            stack_.push(unite(automata));
                        }
                        else
                            std::cerr << "unite: not enough automata on stack" << std::endl;
                        break;
                    }
                    case op_fuse: {
                        long int n = 2;
                        if (specs.has("n")) {
                            char* error;
                            n = strtol(specs["n"].c_str(), &error, 10);
                            if ((*error != '\0') || (n < 2)) {
                                std::cerr << "fuse: n must be larger than 1" << std::endl;
                                break;
                            }
                        }
                        if (stack_.size() >= size_t(n)) {
                            Core::Vector<ConstAutomatonRef> automata;
                            for (u32 i = 0; i < u32(n); ++i)
                                automata.push_back(stack_.pop());
                            stack_.push(fuse(automata));
                        }
                        else
                            std::cerr << "fuse: not enough automata on stack" << std::endl;
                        break;
                    }
                    case op_draw: {
                        oneOperand();
                        std::string file = "-";
                        if (specs.has("file"))
                            file = specs["file"];
                        Core::CompressedOutputStream dos(file);
                        Fsa::Hint                    hints = Fsa::HintNone;
                        if (specs.has("best"))
                            hints |= Fsa::HintMarkBest;
                        if (specs.has("detailed"))
                            hints |= Fsa::HintShowDetails;
                        if (specs.has("linear"))
                            hints |= Fsa::HintAsProbability;
                        Fsa::drawDot(first_, dos, hints, progress);
                        stack_.push(first_);
                        break;
                    }
                    case op_write: {
                        oneOperand();
                        std::string file = "-";
                        if (specs.has("file"))
                            file = specs["file"];
                        if (specs.has("input") && specs.has("output"))
                            write(first_, file, storeAlphabets, progress);
                        else if (specs.has("input"))
                            writeAlphabet(first_->getInputAlphabet(), file);
                        else if (specs.has("output"))
                            writeAlphabet(first_->getOutputAlphabet(), file);
                        else if (specs.has("states"))
                            write(first_, file, storeStates, progress);
                        else
                            write(first_, file, storeAll, progress);
                        stack_.push(first_);
                        break;
                    }

                    case op_cache:
                        oneOperand();
                        stack_.push(cache(first_));
                        break;
                    case op_copy: {
                        oneOperand();
                        if (specs.has("compact"))
                            stack_.push(staticCompactCopy(first_));
                        else if (specs.has("packed"))
                            stack_.push(packedCopy(first_));
                        else
                            stack_.push(staticCopy(first_));
                        first_.reset();
                        break;
                    }
                    case op_delete:
                        stack_.pop();
                        break;
                    case op_default:
                        if (!specs.has("semiring"))
                            std::cerr << "default: needs identifier" << std::endl;
                        else {
                            semiring_ = getSemiring(SemiringType(SemiringTypeChoice[specs["semiring"]]));
                            if (!semiring_) {
                                std::cerr << "unknown semiring '" << specs["semiring"]
                                          << "'. resetting to tropical semiring." << std::endl;
                                semiring_ = TropicalSemiring;
                            }
                        }
                        break;
                    case op_semiring:
                        oneOperand();
                        if (!specs.has("semiring"))
                            std::cerr << "semiring: needs identifier" << std::endl;
                        else {
                            SemiringType     type = SemiringType(SemiringTypeChoice[specs["semiring"]]);
                            ConstSemiringRef semiring;
                            if (type == SemiringTypeLog) {
                                if (specs.has("tolerance")) {
                                    char*    error;
                                    long int tolerance = strtol(specs["tolerance"].c_str(), &error, 10);
                                    if ((*error == '\0') && (tolerance >= 0))
                                        semiring = ConstSemiringRef(new LogSemiring_(tolerance));
                                    else {
                                        std::cerr << "best: n must be a positive integer" << std::endl;
                                        semiring = getSemiring(type);
                                    }
                                }
                                else
                                    semiring = getSemiring(type);
                            }
                            else
                                semiring = getSemiring(type);
                            if (semiring)
                                stack_.push(changeSemiring(first_, semiring));
                        }
                        break;

                    case op_count:
                        oneOperand();
                        if (specs.has("paths"))
                            std::cout << countPaths(first_) << std::endl;
                        else {
                            if (!specs.has("label"))
                                std::cerr << "count: needs [label]" << std::endl;
                            else {
                                size_t count = 0;
                                if (specs.has("input"))
                                    count = countInput(first_, first_->getInputAlphabet()->index(specs["label"]));
                                else
                                    count = countOutput(first_, first_->getOutputAlphabet()->index(specs["label"]));
                                std::cout << count << std::endl;
                            }
                        }
                        stack_.push(first_);
                        break;
                    case op_info:
                        oneOperand();
                        if (specs.has("cheap"))
                            cheapInfo(first_, log());
                        else
                            info(first_, log(), progress);
                        stack_.push(first_);
                        break;
                    case op_memory:
                        oneOperand();
                        memoryInfo(first_, log());
                        stack_.push(first_);
                        break;
                    case op_time:
                        if (timer[0])
                            log() << *(timer[0]);
                        break;
                    case op_wait:
                        std::cerr << "press <ENTER> to continue" << std::flush;
                        getchar();
                        break;
                    default:
                        break;
                }
            }
            timer[1]->stop();
            if (timer[0])
                delete timer[0];
            timer[0] = timer[1];
            timer[1] = 0;
        }
        if (timer[0])
            delete timer[0];
        return EXIT_SUCCESS;
    }
};

APPLICATION(FsaTool)

Core::Choice FsaTool::OperationChoice("best", op_best,
                                      "closure", op_closure,
                                      "collect", op_collect,
                                      "concat", op_concat,
                                      "complement", op_complement,
                                      "compose", op_compose,
                                      "determinize", op_determinize,
                                      "difference", op_difference,
                                      "duplicate", op_duplicate,
                                      "expm", op_expm,
                                      "extend", op_extend,
                                      "invert", op_invert,
                                      "levenshtein", op_levenshtein,
                                      "map-input", op_map_input,
                                      "map-output", op_map_output,
                                      "minimize", op_minimize,
                                      "multiply", op_multiply,
                                      "normalize", op_normalize,
                                      "partial", op_partial,
                                      "permute", op_permute,
                                      "posterior", op_posterior,
                                      "posterior64", op_posterior_64,
                                      "posteriorE", op_posterior_expectation,
                                      "posteriorFail", op_posterior_fail,
                                      "project", op_project,
                                      "prune", op_prune,
                                      "push", op_push,
                                      "random", op_random,
                                      "remove", op_remove,
                                      "sort", op_sort,
                                      "sync-prune", op_sync_prune,
                                      "transpose", op_transpose,
                                      "trim", op_trim,
                                      "unite", op_unite,
                                      "fuse", op_fuse,

                                      "draw", op_draw,
                                      "write", op_write,

                                      "cache", op_cache,
                                      "copy", op_copy,
                                      "delete", op_delete,
                                      "default", op_default,
                                      "semiring", op_semiring,

                                      "count", op_count,
                                      "info", op_info,
                                      "memory", op_memory,
                                      "time", op_time,
                                      "wait", op_wait,
                                      Core::Choice::endMark());

Core::ParameterBool FsaTool::paramProgress("progress", "show progress of operations", false);
