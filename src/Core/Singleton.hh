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
#ifndef _CORE_SINGLETON_HH
#define _CORE_SINGLETON_HH

namespace Core {
/**
 * Implementation of the Singleton Pattern.
 * Inspired by A. Alexandrescu "Modern C++ Design"
 */
template<class T>
class SingletonHolder {
public:
    typedef T        Instance;
    static Instance& instance();

private:
    SingletonHolder();
    SingletonHolder(const SingletonHolder&);

    struct StaticHolder {
        Instance* instance_;  // will be inited with zero (because it's static)
        ~StaticHolder() {
            delete instance_;
        }
    };

    static StaticHolder staticHolder_;
};

template<class T>
typename SingletonHolder<T>::StaticHolder SingletonHolder<T>::staticHolder_;

template<class T>
inline typename SingletonHolder<T>::Instance& SingletonHolder<T>::instance() {
    if (!staticHolder_.instance_) {
        staticHolder_.instance_ = new Instance;
    }
    return *staticHolder_.instance_;
}

}  // namespace Core
#endif /* _CORE_SINGLETON_HH */
