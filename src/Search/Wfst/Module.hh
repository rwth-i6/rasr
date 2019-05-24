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
#ifndef _SEARCH_MODULE_HH
#define _SEARCH_MODULE_HH
#include <Core/Factory.hh>
#include <Core/Singleton.hh>
#include <Search/Wfst/Builder.hh>

namespace Search {
namespace Wfst {

class Module_ {
public:
    Module_();

    class BuilderFactory : public Core::Factory<Builder::Operation,
                                                Builder::Operation* (*)(const Core::Configuration&, Builder::Resources&),
                                                std::string> {
    public:
        typedef Builder::Operation* Result;
        typedef Result (*CreationFunction)(const Core::Configuration&, Builder::Resources&);
        typedef std::string Identifier;

    public:
        Result getObject(const Identifier& id, const Core::Configuration& c, Builder::Resources& r) const {
            CreationFunction create = getCreationFunction(id);
            if (create)
                return create(c, r);
            else
                return 0;
        }
        template<class T>
        static Result create(const Core::Configuration& c, Builder::Resources& r) {
            return new T(c, r);
        }
    };

private:
    BuilderFactory builderFactory_;
    template<class T>
    void registerBuilderOperation() {
        builderFactory_.registerClass(T::name(), BuilderFactory::create<T>);
    }

public:
    Builder::Operation* getBuilderOperation(const std::string&         id,
                                            const Core::Configuration& c, Builder::Resources& r) const {
        return builderFactory_.getObject(id, c, r);
    }

    std::vector<std::string> builderOperations() const {
        return builderFactory_.identifiers();
    }
};

typedef Core::SingletonHolder<Module_> Module;

}  // namespace Wfst
}  // namespace Search

#endif /* _SEARCH_MODULE_HH */
