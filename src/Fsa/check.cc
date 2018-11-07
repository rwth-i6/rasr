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
// $Id$

/**
 * This program illustrates typical use of some SprintCore functions.
 */

#include "Application.hh"
#include "Automaton.hh"
#include "Basic.hh"
#include "Cache.hh"
#include "Input.hh"
#include "Output.hh"
#include "Resources.hh"
#include "Static.hh"

// ===========================================================================
// Application

        /**
         * queue:
         * - end of queue is defined as self-referencing state id
         **/
        class SsspQueue {
        protected:
                Fsa::StateId head_;
                Core::Vector<Fsa::StateId> queue_;
        public:
                SsspQueue() :
                        head_(Fsa::InvalidStateId) {
                }
                bool empty() const {
                        return head_ == Fsa::InvalidStateId;
                }
                Fsa::StateId dequeue() {
                        require_(!empty());
                        Fsa::StateId s = head_;
                        head_ = queue_[s];
                        if (head_ == s)
                                head_ = Fsa::InvalidStateId;
                        queue_[s] = Fsa::InvalidStateId;
                        return s;
                }
                Fsa::StateId maxStateId() const {
                        return Fsa::InvalidStateId;
                }
        };

        class FifoSsspQueue : public SsspQueue {
        public:
                void enqueue(Fsa::StateId s) {
                        queue_.grow(s, Fsa::InvalidStateId);
                        if (queue_[s] == Fsa::InvalidStateId) {
                                if (head_ != Fsa::InvalidStateId)
                                        queue_[s] = head_;
                                else
                                        queue_[s] = s;
                                head_ = s;
                        }
                }
        };


class MyApplication : public Fsa::Application {
public:
    std::string getUsage() const {
        return "...";
    }

    int main(const std::vector<std::string> &arguments) {
#if 0
        Fsa::Resources &r = Fsa::getResources();
        r.dump(std::cerr);

        if (arguments.empty()) return 0;
        Fsa::ConstAutomatonRef f = Fsa::cache(Fsa::trim(Fsa::read(arguments[0])));
        Fsa::writeAtt(f, std::cout);
        std::cout << std::endl;

        Fsa::StringPotentialsRef p =Fsa::stringPotentials(f);
        p->dump(std::cout);
        Fsa::ConstAutomatonRef pf, ef;
        pf = Fsa::pushOutputToInitial(f, p);
        std::cout << std::endl;
        Fsa::writeAtt(pf, std::cout);
        pf = Fsa::pushAndPullOutputToInitial(f, p);
        std::cout << std::endl;
        Fsa::writeAtt(pf, std::cout);
        ef = Fsa::removeRedundantEpsilons(pf);
        std::cout << std::endl;
        Fsa::writeAtt(ef, std::cout);

        /*
        f = Fsa::pullOutput(f);
        std::cout << std::endl;
        std::cout << std::endl;
        Fsa::writeAtt(f, std::cout);
        */

        /*
        f = Fsa::removeRedundantEpsilons(f);
        std::cout << std::endl;
        std::cout << std::endl;
        Fsa::writeAtt(f, std::cout);
        */
#endif
#if 0
        Core::Ref<Fsa::StaticAutomaton> s = Fsa::staticCopy("test fsa");
        Fsa::ConstAutomatonRef f(s);
        Fsa::write(f, "-");
#endif


        FifoSsspQueue q;
        for(Fsa::StateId i=1; i<20; ++i)
                q.enqueue(i);
        while(!q.empty()) {
                Fsa::StateId s = q.dequeue();
                std::cout << s << std::endl;
        }

        return 0;
    }
} app; // <- You have to create ONE instance of the application

APPLICATION
