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
#ifndef _SEARCH_IO_OPERATIONS_HH
#define _SEARCH_IO_OPERATIONS_HH

#include <Search/Wfst/Builder.hh>

namespace Search {
namespace Wfst {
namespace Builder {

/**
 * base class for file operations
 */
class FileOperation : public virtual Operation {
protected:
    static const Core::ParameterString paramFilename;
    static const Core::Choice          choiceType;
    static const Core::ParameterChoice paramType;
    enum FileType { TypeVector,
                    TypeConst,
                    TypeCompact,
                    TypeNGram };
    std::string filename() const;

public:
    FileOperation(const Core::Configuration& c, Resources& r)
            : Operation(c, r) {}

protected:
    virtual bool precondition() const;
};

class ReadOperation : public FileOperation {
public:
    ReadOperation(const Core::Configuration& c, Resources& r)
            : Operation(c, r), FileOperation(c, r) {}

protected:
    static const Core::ParameterStringVector paramAttributes;
    void                                     attachAttributes(AutomatonRef automaton) const;
};

/**
 * base class for read operations
 */
class ReadFst : public ReadOperation {
public:
    ReadFst(const Core::Configuration& c, Resources& r)
            : Operation(c, r), ReadOperation(c, r) {}

protected:
    virtual AutomatonRef process();

public:
    static std::string name() {
        return "read-fst";
    }
};

/**
 * read automaton in Fsa format
 */
class ReadFsa : public ReadOperation {
public:
    ReadFsa(const Core::Configuration& c, Resources& r)
            : Operation(c, r), ReadOperation(c, r) {}

protected:
    virtual AutomatonRef process();

public:
    static std::string name() {
        return "read-fsa";
    }
};

/**
 * base class for write operations
 */
class WriteOperation : public SleeveOperation, protected FileOperation {
public:
    WriteOperation(const Core::Configuration& c, Resources& r)
            : Operation(c, r), SleeveOperation(c, r), FileOperation(c, r) {}

protected:
    virtual bool precondition() const;
};

/**
 * write automaton in Fst format
 */
class WriteFst : public WriteOperation {
public:
    WriteFst(const Core::Configuration& c, Resources& r)
            : Operation(c, r), WriteOperation(c, r) {}

protected:
    virtual AutomatonRef process();

private:
    template<class F>
    bool convertAndWrite(const std::string& filename) const;

public:
    static std::string name() {
        return "write-fst";
    }
};

/**
 * write automaton in Fsa format
 */
class WriteFsa : public WriteOperation {
public:
    WriteFsa(const Core::Configuration& c, Resources& r)
            : Operation(c, r), WriteOperation(c, r) {}

protected:
    virtual AutomatonRef process();

public:
    static std::string name() {
        return "write-fsa";
    }
};

/**
 * write automaton in a compressed format, to be used with Search::CompressedNetwork
 * disk usage is about the same as with OpenFst::VectorFst format,
 * but the size in memory is lower.
 */
class Compress : public WriteOperation {
public:
    Compress(const Core::Configuration& c, Resources& r)
            : Operation(c, r), WriteOperation(c, r) {}

protected:
    virtual AutomatonRef process();

public:
    static std::string name() {
        return "compress";
    }
};

}  // namespace Builder
}  // namespace Wfst
}  // namespace Search

#endif  // _SEARCH_IO_OPERATIONS_HH
