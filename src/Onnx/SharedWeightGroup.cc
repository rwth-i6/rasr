/** Copyright 2026 RWTH Aachen University. All rights reserved.
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

#include "SharedWeightGroup.hh"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#ifdef MODULE_CUDA
#include <cuda_runtime.h>
#endif

namespace {

using ByteView = std::pair<uint8_t const*, size_t>;

struct ProtoReader {
    uint8_t const* cur;
    uint8_t const* end;

    explicit ProtoReader(ByteView view)
            : cur(view.first), end(view.first + view.second) {}

    bool eof() const {
        return cur == end;
    }

    uint64_t readVarint() {
        uint64_t value = 0;
        for (int shift = 0; shift < 64; shift += 7) {
            if (cur == end) {
                throw std::runtime_error("unexpected end of protobuf varint");
            }
            uint8_t byte = *cur++;
            value |= uint64_t(byte & 0x7f) << shift;
            if ((byte & 0x80) == 0) {
                return value;
            }
        }
        throw std::runtime_error("invalid protobuf varint");
    }

    ByteView readBytes() {
        uint64_t length = readVarint();
        if (static_cast<uint64_t>(end - cur) < length) {
            throw std::runtime_error("unexpected end of protobuf bytes field");
        }
        ByteView view(cur, static_cast<size_t>(length));
        cur += length;
        return view;
    }

    void skip(uint32_t wireType) {
        switch (wireType) {
            case 0:
                readVarint();
                break;
            case 1:
                requireBytes(8);
                cur += 8;
                break;
            case 2:
                readBytes();
                break;
            case 5:
                requireBytes(4);
                cur += 4;
                break;
            default:
                throw std::runtime_error("unsupported protobuf wire type");
        }
    }

    void requireBytes(size_t n) const {
        if (static_cast<size_t>(end - cur) < n) {
            throw std::runtime_error("unexpected end of protobuf fixed field");
        }
    }
};

struct MappedData {
    uint8_t const* data = nullptr;
    size_t         size = 0;

    virtual ~MappedData() = default;

    // Hint that the data will be read sequentially soon, to get large read-ahead from (network) file systems
    virtual void willNeed() const {}
    // Drop the resident pages, they are read again from the file (or page cache) on the next access
    virtual void dontNeed() const {}
};

struct ParsedInitializer {
    std::string                 name;
    ONNXTensorElementDataType   dataType = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    std::vector<int64_t>        shape;
    std::shared_ptr<MappedData> data;
};

struct MappedFileData : MappedData {
    void*  mapping       = MAP_FAILED;
    size_t mappingLength = 0;

    MappedFileData(std::filesystem::path const& path, uint64_t offset, std::optional<uint64_t> length) {
        int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) {
            throw std::runtime_error("failed to open ONNX data file " + path.string());
        }
        // The mapping keeps its own reference to the file, so the descriptor is not needed afterwards
        try {
            map(fd, path, offset, length);
        }
        catch (...) {
            close(fd);
            throw;
        }
        close(fd);
    }

    void willNeed() const override {
        madvise(mapping, mappingLength, MADV_SEQUENTIAL);
        madvise(mapping, mappingLength, MADV_WILLNEED);
    }

    void dontNeed() const override {
        madvise(mapping, mappingLength, MADV_NORMAL);  // later accesses by ONNX Runtime are not necessarily sequential
        madvise(mapping, mappingLength, MADV_DONTNEED);
    }

    ~MappedFileData() override {
        if (mapping != MAP_FAILED) {
            munmap(mapping, mappingLength);
        }
    }

private:
    void map(int fd, std::filesystem::path const& path, uint64_t offset, std::optional<uint64_t> length) {
        struct stat statBuffer;
        if (fstat(fd, &statBuffer) != 0 or statBuffer.st_size < 0) {
            throw std::runtime_error("failed to determine size of ONNX data file " + path.string());
        }

        uint64_t fileSize = static_cast<uint64_t>(statBuffer.st_size);
        if (offset > fileSize) {
            throw std::runtime_error("invalid offset in ONNX data file " + path.string());
        }

        uint64_t dataLength = length.value_or(fileSize - offset);
        if (offset + dataLength > fileSize) {
            throw std::runtime_error("invalid length in ONNX data file " + path.string());
        }

        long pageSize = sysconf(_SC_PAGESIZE);
        if (pageSize <= 0) {
            throw std::runtime_error("failed to determine page size for mmap");
        }

        uint64_t pageOffset = offset % static_cast<uint64_t>(pageSize);
        uint64_t mapOffset  = offset - pageOffset;
        mappingLength       = static_cast<size_t>(dataLength + pageOffset);
        mapping             = mmap(nullptr, mappingLength, PROT_READ, MAP_PRIVATE, fd, static_cast<off_t>(mapOffset));
        if (mapping == MAP_FAILED) {
            throw std::runtime_error("failed to mmap ONNX data file " + path.string());
        }

        data = static_cast<uint8_t const*>(mapping) + pageOffset;
        size = static_cast<size_t>(dataLength);
    }
};

// Tensor data embedded in the model file itself (raw_data)
struct MappedSubrangeData : MappedData {
    std::shared_ptr<MappedData> owner;

    MappedSubrangeData(std::shared_ptr<MappedData> owner, uint8_t const* data, size_t size)
            : owner(std::move(owner)) {
        this->data = data;
        this->size = size;
    }

    void willNeed() const override {
        advise(MADV_SEQUENTIAL);
        advise(MADV_WILLNEED);
    }

    void dontNeed() const override {
        advise(MADV_NORMAL);
        advise(MADV_DONTNEED);
    }

private:
    // madvise needs a page-aligned start. Affecting the neighboring data in the boundary pages is harmless.
    void advise(int advice) const {
        uintptr_t pageSize = static_cast<uintptr_t>(sysconf(_SC_PAGESIZE));
        uintptr_t begin    = reinterpret_cast<uintptr_t>(data) / pageSize * pageSize;
        uintptr_t end      = reinterpret_cast<uintptr_t>(data) + size;
        madvise(reinterpret_cast<void*>(begin), end - begin, advice);
    }
};

std::shared_ptr<MappedData> mapFile(std::filesystem::path const& path, uint64_t offset = 0, std::optional<uint64_t> length = std::nullopt) {
    return std::make_shared<MappedFileData>(path, offset, length);
}

std::string bytesToString(ByteView view) {
    return std::string(reinterpret_cast<char const*>(view.first), view.second);
}

size_t elementSize(ONNXTensorElementDataType type) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32:
            return 4;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
            return 1;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
            return 2;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64:
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64:
            return 8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128:
            return 16;
        default:
            return 0;
    }
}

size_t tensorByteSize(ParsedInitializer const& initializer) {
    size_t elems = 1;
    for (int64_t dim : initializer.shape) {
        if (dim < 0) {
            throw std::runtime_error("initializer " + initializer.name + " has a negative dimension");
        }
        elems *= static_cast<size_t>(dim);
    }

    size_t elemSize = elementSize(initializer.dataType);
    if (elemSize == 0) {
        return 0;
    }
    return elems * elemSize;
}

std::unordered_map<std::string, std::string> parseExternalData(ByteView view) {
    std::unordered_map<std::string, std::string> result;
    ProtoReader                                  reader(view);
    std::string                                  key;
    std::string                                  value;

    while (not reader.eof()) {
        uint64_t tag      = reader.readVarint();
        uint32_t field    = static_cast<uint32_t>(tag >> 3);
        uint32_t wireType = static_cast<uint32_t>(tag & 0x7);
        if (wireType == 2 and field == 1) {
            key = bytesToString(reader.readBytes());
        }
        else if (wireType == 2 and field == 2) {
            value = bytesToString(reader.readBytes());
        }
        else {
            reader.skip(wireType);
        }
    }

    if (not key.empty()) {
        result.emplace(std::move(key), std::move(value));
    }
    return result;
}

std::optional<ParsedInitializer> parseTensor(ByteView view, std::filesystem::path const& modelDir, std::shared_ptr<MappedData> const& modelData) {
    ProtoReader                                  reader(view);
    ParsedInitializer                            initializer;
    ByteView                                     rawData(nullptr, 0);
    std::unordered_map<std::string, std::string> externalData;

    while (not reader.eof()) {
        uint64_t tag      = reader.readVarint();
        uint32_t field    = static_cast<uint32_t>(tag >> 3);
        uint32_t wireType = static_cast<uint32_t>(tag & 0x7);

        if (field == 1 and wireType == 0) {
            initializer.shape.push_back(static_cast<int64_t>(reader.readVarint()));
        }
        else if (field == 1 and wireType == 2) {
            ProtoReader packed(reader.readBytes());
            while (not packed.eof()) {
                initializer.shape.push_back(static_cast<int64_t>(packed.readVarint()));
            }
        }
        else if (field == 2 and wireType == 0) {
            initializer.dataType = static_cast<ONNXTensorElementDataType>(reader.readVarint());
        }
        else if (field == 8 and wireType == 2) {
            initializer.name = bytesToString(reader.readBytes());
        }
        else if (field == 9 and wireType == 2) {
            rawData = reader.readBytes();
        }
        else if (field == 13 and wireType == 2) {
            auto entry = parseExternalData(reader.readBytes());
            externalData.insert(entry.begin(), entry.end());
        }
        else {
            reader.skip(wireType);
        }
    }

    if (initializer.name.empty() or initializer.dataType == ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED) {
        return std::nullopt;
    }

    size_t expectedBytes = tensorByteSize(initializer);
    if (expectedBytes == 0) {
        return std::nullopt;
    }

    if (rawData.first != nullptr) {
        initializer.data = std::make_shared<MappedSubrangeData>(modelData, rawData.first, rawData.second);
    }
    else if (auto locationIter = externalData.find("location"); locationIter != externalData.end()) {
        uint64_t                offset = 0;
        std::optional<uint64_t> length;
        if (auto iter = externalData.find("offset"); iter != externalData.end()) {
            offset = std::stoull(iter->second);
        }
        if (auto iter = externalData.find("length"); iter != externalData.end()) {
            length = std::stoull(iter->second);
        }
        initializer.data = mapFile(modelDir / locationIter->second, offset, length);
    }
    else {
        return std::nullopt;
    }

    if (initializer.data->size != expectedBytes) {
        throw std::runtime_error("initializer " + initializer.name + " has " + std::to_string(initializer.data->size) +
                                 " bytes but shape and dtype imply " + std::to_string(expectedBytes));
    }

    return initializer;
}

void parseGraph(ByteView view, std::filesystem::path const& modelDir, std::shared_ptr<MappedData> const& modelData, std::vector<ParsedInitializer>& initializers) {
    ProtoReader reader(view);
    while (not reader.eof()) {
        uint64_t tag      = reader.readVarint();
        uint32_t field    = static_cast<uint32_t>(tag >> 3);
        uint32_t wireType = static_cast<uint32_t>(tag & 0x7);
        if (field == 5 and wireType == 2) {
            auto initializer = parseTensor(reader.readBytes(), modelDir, modelData);
            if (initializer) {
                initializers.push_back(std::move(*initializer));
            }
        }
        else {
            reader.skip(wireType);
        }
    }
}

std::vector<ParsedInitializer> parseModelInitializers(std::string const& modelFile) {
    std::filesystem::path modelPath(modelFile);
    auto                  modelData = mapFile(modelPath);

    std::vector<ParsedInitializer> initializers;
    ProtoReader                    reader(ByteView(modelData->data, modelData->size));
    while (not reader.eof()) {
        uint64_t tag      = reader.readVarint();
        uint32_t field    = static_cast<uint32_t>(tag >> 3);
        uint32_t wireType = static_cast<uint32_t>(tag & 0x7);
        if (field == 7 and wireType == 2) {
            parseGraph(reader.readBytes(), modelPath.parent_path(), modelData, initializers);
        }
        else {
            reader.skip(wireType);
        }
    }
    return initializers;
}

/*
 * ONNX Runtime allocates initializers with this alignment and its CPU kernels may rely on it. Data in external data
 * files usually starts at a page boundary, but tensor data embedded in the model file (raw_data) has arbitrary offsets.
 */
constexpr size_t initializerAlignment = 64ul;

// Aligned copy of tensor data, used on CPU for data which cannot be used in place
struct AlignedCopyData : MappedData {
    explicit AlignedCopyData(MappedData const& source) {
        size_t allocationSize = (source.size + initializerAlignment - 1ul) / initializerAlignment * initializerAlignment;
        void*  buffer         = std::aligned_alloc(initializerAlignment, allocationSize);
        if (buffer == nullptr) {
            throw std::runtime_error("failed to allocate " + std::to_string(allocationSize) + " bytes for shared initializer");
        }
        std::memcpy(buffer, source.data, source.size);
        data = static_cast<uint8_t const*>(buffer);
        size = source.size;
    }

    AlignedCopyData(AlignedCopyData const&)            = delete;
    AlignedCopyData& operator=(AlignedCopyData const&) = delete;

    ~AlignedCopyData() override {
        std::free(const_cast<uint8_t*>(data));
    }
};

#ifdef MODULE_CUDA
struct CudaBuffer {
    void* ptr = nullptr;

    CudaBuffer(void const* hostData, size_t size) {
        if (cudaMalloc(&ptr, size) != cudaSuccess) {
            ptr = nullptr;
            throw std::runtime_error("failed to allocate " + std::to_string(size) + " bytes of CUDA memory for shared initializer");
        }
        if (cudaMemcpy(ptr, hostData, size, cudaMemcpyHostToDevice) != cudaSuccess) {
            cudaFree(ptr);
            ptr = nullptr;
            throw std::runtime_error("failed to copy shared initializer to CUDA device");
        }
    }

    CudaBuffer(CudaBuffer const&)            = delete;
    CudaBuffer& operator=(CudaBuffer const&) = delete;

    ~CudaBuffer() {
        if (ptr != nullptr) {
            cudaFree(ptr);
        }
    }
};
#endif

/*
 * One deduplicated initializer tensor. On CPU the tensor points directly into the memory-mapped external data file
 * (data stored inside the model file is usually not aligned and therefore copied), on CUDA it owns a device copy.
 * The host data is kept in both cases to compare it with the initializers of further models.
 */
struct SharedInitializer {
    ONNXTensorElementDataType   dataType;
    std::vector<int64_t>        shape;
    size_t                      bytes;
    uint64_t                    hash;
    std::shared_ptr<MappedData> hostData;
#ifdef MODULE_CUDA
    std::unique_ptr<CudaBuffer> deviceData;
#endif
    Ort::Value value;  // declared last so it is released before the buffer it refers to

    SharedInitializer(ParsedInitializer&& initializer, uint64_t hash, Ort::MemoryInfo const& memoryInfo, bool onCuda)
            : dataType(initializer.dataType),
              shape(std::move(initializer.shape)),
              bytes(initializer.data->size),
              hash(hash),
              hostData(std::move(initializer.data)),
              value(nullptr) {
        void* buffer = nullptr;
        if (onCuda) {
#ifdef MODULE_CUDA
            deviceData = std::make_unique<CudaBuffer>(hostData->data, bytes);
            buffer     = deviceData->ptr;
#else
            throw std::runtime_error("CUDA shared initializers require RASR to be compiled with MODULE_CUDA");
#endif
        }
        else {
            if (reinterpret_cast<uintptr_t>(hostData->data) % initializerAlignment != 0ul) {
                hostData = std::make_shared<AlignedCopyData>(*hostData);
            }
            buffer = const_cast<uint8_t*>(hostData->data);
        }
        value = Ort::Value::CreateTensor(memoryInfo, buffer, bytes, shape.data(), shape.size(), dataType);
    }
};

/*
 * Small initializers are not worth sharing. Additionally, with the CUDA execution provider small
 * integer tensors are typically shape information consumed on the CPU, so a CUDA copy would not be
 * used by ONNX Runtime anyway.
 */
constexpr size_t minSharedInitializerBytes = 64ul * 1024ul;

bool isShareable(ParsedInitializer const& initializer, bool onCuda) {
    if (initializer.data->size < minSharedInitializerBytes) {
        return false;
    }
    if (onCuda) {
        switch (initializer.dataType) {
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
            case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL:
                return false;
            default:
                break;
        }
    }
    return true;
}

// Only used to find candidates, equality is always confirmed by comparing the data
uint64_t hashData(MappedData const& data) {
    // FNV-1a over 64 bit words with a final mixing step, much faster than byte-wise hashing of multi-GB models
    constexpr uint64_t prime = 1099511628211ull;
    uint64_t           hash  = 1469598103934665603ull ^ data.size;
    size_t             i     = 0ul;
    for (; i + sizeof(uint64_t) <= data.size; i += sizeof(uint64_t)) {
        uint64_t word;
        std::memcpy(&word, data.data + i, sizeof(word));
        hash = (hash ^ word) * prime;
    }
    for (; i < data.size; ++i) {
        hash = (hash ^ data.data[i]) * prime;
    }
    hash ^= hash >> 33;
    hash *= 0xff51afd7ed558ccdull;
    hash ^= hash >> 33;
    return hash;
}

bool isEqual(SharedInitializer const& shared, ParsedInitializer const& parsed, uint64_t hash) {
    return shared.hash == hash and shared.dataType == parsed.dataType and shared.shape == parsed.shape and
           shared.bytes == parsed.data->size and std::memcmp(shared.hostData->data, parsed.data->data, shared.bytes) == 0;
}

bool isCuda(Onnx::ExecutionProviderType provider) {
    return provider == Onnx::ExecutionProviderType::cuda;
}

Ort::MemoryInfo createMemoryInfo(Onnx::ExecutionProviderType provider) {
    if (isCuda(provider)) {
        // Must match the device of the CUDA execution provider, which uses its default device 0
        return Ort::MemoryInfo("Cuda", OrtDeviceAllocator, 0, OrtMemTypeDefault);
    }
    return Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeDefault);
}

}  // namespace

namespace Onnx {

struct SharedWeightGroup::Impl {
    ExecutionProviderType                                                         provider;
    Ort::MemoryInfo                                                               memoryInfo;
    Ort::PrepackedWeightsContainer                                                prepackedWeightsContainer;
    std::mutex                                                                    mutex;
    std::unordered_map<uint64_t, std::vector<std::shared_ptr<SharedInitializer>>> initializersByHash;
    size_t                                                                        totalBytes;

    Impl(ExecutionProviderType provider)
            : provider(provider),
              memoryInfo(createMemoryInfo(provider)),
              prepackedWeightsContainer(),
              mutex(),
              initializersByHash(),
              totalBytes(0ul) {}
};

std::shared_ptr<SharedWeightGroup> SharedWeightGroup::get(std::string const& name, ExecutionProviderType provider) {
    static std::mutex                                                        mutex;
    static std::unordered_map<std::string, std::weak_ptr<SharedWeightGroup>> groups;

    std::lock_guard<std::mutex> lock(mutex);
    auto                        group = groups[name].lock();
    if (not group) {
        group        = std::shared_ptr<SharedWeightGroup>(new SharedWeightGroup(provider));
        groups[name] = group;
    }
    else if (group->impl_->provider != provider) {
        throw std::runtime_error("weight group '" + name + "' is already used by sessions with a different execution provider");
    }
    return group;
}

SharedWeightGroup::SharedWeightGroup(ExecutionProviderType provider)
        : impl_(std::make_unique<Impl>(provider)) {}

SharedWeightGroup::~SharedWeightGroup() = default;

OrtPrepackedWeightsContainer* SharedWeightGroup::prepackedWeightsContainer() {
    return impl_->prepackedWeightsContainer;
}

SharedWeightGroup::Statistics SharedWeightGroup::addInitializers(std::string const& modelFile, Ort::SessionOptions& sessionOptions) {
    auto parsedInitializers = parseModelInitializers(modelFile);
    bool onCuda             = isCuda(impl_->provider);

    std::lock_guard<std::mutex>     lock(impl_->mutex);
    Statistics                      statistics;
    std::unordered_set<std::string> addedNames;
    for (auto& parsed : parsedInitializers) {
        if (not isShareable(parsed, onCuda) or not addedNames.insert(parsed.name).second) {
            continue;
        }

        parsed.data->willNeed();
        uint64_t                           hash       = hashData(*parsed.data);
        auto&                              candidates = impl_->initializersByHash[hash];
        std::shared_ptr<SharedInitializer> initializer;
        for (auto const& candidate : candidates) {
            if (isEqual(*candidate, parsed, hash)) {
                initializer = candidate;
                break;
            }
        }

        std::string name = parsed.name;
        if (initializer) {
            statistics.sharedInitializers += 1ul;
            statistics.sharedBytes += initializer->bytes;
            // Release the duplicate right away instead of keeping all mappings of this model until the end
            parsed.data.reset();
        }
        else {
            initializer = std::make_shared<SharedInitializer>(std::move(parsed), hash, impl_->memoryInfo, onCuda);
            candidates.push_back(initializer);
            impl_->totalBytes += initializer->bytes;
            statistics.newInitializers += 1ul;
            statistics.newBytes += initializer->bytes;
            // ONNX Runtime reads the data again when it needs it, don't keep it resident in between
            initializer->hostData->dontNeed();
        }
        sessionOptions.AddInitializer(name.c_str(), initializer->value);
    }
    statistics.groupBytes = impl_->totalBytes;

    return statistics;
}

void SharedWeightGroup::releaseHostPages() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    for (auto const& [hash, initializers] : impl_->initializersByHash) {
        for (auto const& initializer : initializers) {
            initializer->hostData->dontNeed();
        }
    }
}

}  // namespace Onnx
