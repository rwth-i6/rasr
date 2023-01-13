#include "AlignedFeatureCache.hh"
#include <Core/BinaryStream.hh>
#include <Core/CompressedStream.hh>
#include <Core/Directory.hh>
#include <Flow/DataAdaptor.hh>
#include <Flow/Registry.hh>
#include "Alignment.hh"
#include "ModelCombination.hh"

using namespace Speech;

SortedCacheWriter::SortedCacheWriter(u32 nIds, const std::string& cacheDir, u32 bufferSize)
        : SortedCache(cacheDir, bufferSize) {
    initialize(nIds, bufferSize);
}

Speech::SortedCacheWriter::~SortedCacheWriter() {
    finish();
}

void SortedCacheWriter::initialize(u32 nIds, u32 bufferSize) {
    bufferSize_ = bufferSize;
    featureCaches_.resize(nIds);
    u32  id          = 0;
    bool cacheExists = false;
    for (std::vector<FeatureBuffer>::iterator c = featureCaches_.begin(); c != featureCaches_.end(); ++c, ++id) {
        c->reserve(bufferSize_);
        if (!cacheExists && Core::isValidPath(cacheFile(id)))
            cacheExists = true;
    }
    if (cacheExists) {
        Core::Application::us()->warning("aligned feature cache already exists. features will be appended");
    }
    else {
        if (!Core::isDirectory(cacheDir_))
            if (!Core::createDirectory(cacheDir_))
                Core::Application::us()->error("cannot create directory %s", cacheDir_.c_str());
    }
}

bool SortedCacheWriter::add(FeaturePtr feature, u32 id) {
    FeatureBuffer& cache = featureCaches_[id];
    verify(feature);
    cache.push_back(feature);
    if (cache.size() >= bufferSize_) {
        if (!writeCache(id))
            return false;
        clearBuffer(id);
    }
    return true;
}

bool SortedCacheWriter::writeCache(u32 id) const {
    Core::BinaryOutputStream out(cacheFile(id), std::ios::app);

    if (!out.good())
        return false;
    writeBuffer(out, id);
    out.close();
    return true;
}

void SortedCacheWriter::writeBuffer(Core::BinaryOutputStream& out, u32 id) const {
    const FeatureBuffer& cache = featureCaches_[id];
    for (FeatureBuffer::const_iterator i = cache.begin(); i != cache.end(); ++i) {
        (*i)->write(out);
    }
}

void SortedCacheWriter::clearBuffer(u32 id) {
    featureCaches_[id].clear();
}

bool SortedCacheWriter::finish() {
    bool ok = true;
    for (u32 id = 0; id < featureCaches_.size(); ++id) {
        if (featureCaches_[id].size() > 0) {
            if (!writeCache(id))
                ok = false;
            clearBuffer(id);
        }
    }
    if (!ok)
        Core::Application::us()->error("could not write all feature caches");
    return ok;
}

// ==========================================

CompressedSortedCacheWriter::CompressedSortedCacheWriter(u32 nIds, const std::string& cacheDir, u32 bufferSize)
        : SortedCacheWriter(nIds, cacheDir, bufferSize) {}

CompressedSortedCacheWriter::~CompressedSortedCacheWriter() {
    finish();
}

void CompressedSortedCacheWriter::initialize(u32 nIds, u32 bufferSize) {
    SortedCacheWriter::initialize(nIds, bufferSize);
    countBuffers_.resize(nIds);
    for (std::vector<CountBuffer>::iterator i = countBuffers_.begin(); i != countBuffers_.end(); ++i)
        i->reserve(bufferSize);
}

bool CompressedSortedCacheWriter::add(FeaturePtr feature, u32 id) {
    const FeatureBuffer& featureBuffer = featureCaches_[id];
    CountBuffer&         countBuffer   = countBuffers_[id];
    bool                 retVal        = true;
    const Flow::Data&    last          = *featureBuffer.back();
    const Flow::Data&    current       = *feature;
    if (featureBuffer.empty() || !(last == current) || countBuffer.back() == Core::Type<u8>::max) {
        countBuffer.push_back(1);
        retVal = SortedCacheWriter::add(feature, id);
    }
    else {
        ++countBuffer.back();
    }
    return retVal;
}

void CompressedSortedCacheWriter::writeBuffer(Core::BinaryOutputStream& out, u32 id) const {
    const FeatureBuffer& featureBuffer = featureCaches_[id];
    const CountBuffer&   countBuffer   = countBuffers_[id];
    verify(featureBuffer.size() == countBuffer.size());
    for (size_t i = 0; i < featureBuffer.size(); ++i) {
        out << countBuffer[i];
        featureBuffer[i]->write(out);
    }
}

void CompressedSortedCacheWriter::clearBuffer(u32 id) {
    SortedCacheWriter::clearBuffer(id);
    countBuffers_[id].clear();
}

// ==========================================

bool Speech::SortedCacheReader::setDatatype(const std::string& datatype) {
    featureType_ = Flow::Registry::instance().getDatatype(datatype);
    return (featureType_ != 0);
}

bool SortedCacheReader::open(u32 id) {
    if (stream_.isOpen())
        close();
    stream_.clear();
    stream_.open(cacheFile(id));
    if (stream_.fail())
        return false;
    stream_.seek(0, std::ios::end);
    if (stream_.fail())
        return false;
    streamEnd_ = stream_.position();
    stream_.seek(0, std::ios::beg);
    if (stream_.fail())
        return false;
    return stream_.good();
}

void SortedCacheReader::close() {
    stream_.close();
}

SortedCacheReader::FeaturePtr SortedCacheReader::getData() {
    if (buffer_.empty()) {
        fillBuffer();
        if (buffer_.empty()) {
            return FeaturePtr();
        }
    }
    FeaturePtr result = buffer_.front();
    buffer_.pop_front();
    return result;
}

void SortedCacheReader::fillBuffer() {
    u32 cnt = 0;
    while (cnt++ < bufferSize_ && stream_.position() < streamEnd_) {
        FeaturePtr ptr(featureType_->newData());
        ptr->read(stream_);
        buffer_.push_back(ptr);
    }
}

// ==========================================

void CompressedSortedCacheReader::fillBuffer() {
    u8 count = 0;
    while (buffer_.size() < bufferSize_ && stream_.position() < streamEnd_) {
        stream_ >> count;
        FeaturePtr ptr(featureType_->newData());
        ptr->read(stream_);
        for (u8 i = 0; i < count; ++i) {
            buffer_.push_back(FeaturePtr(ptr->clone()));
        }
    }
}

// ==========================================

const Core::ParameterString AlignedFeatureCacheWriterNode::paramCacheDirectory(
        "path", "cache directory", ".");
const Core::ParameterBool AlignedFeatureCacheWriterNode::paramCompressed(
        "compressed", "compress caches by counting consecutive equal features", false);
const Core::ParameterInt AlignedFeatureCacheWriterNode::paramBufferSize(
        "buffer-size", "number of feature vectors to buffer", 1024);
const Core::ParameterBool AlignedFeatureCacheWriterNode::paramRepeatFeatures(
        "repeat-features",
        "repeat the last read features if the alignment is longer than the feature stream",
        false);

AlignedFeatureCacheWriterNode::AlignedFeatureCacheWriterNode(const Core::Configuration& config)
        : Core::Component(config),
          Flow::Node(config),
          initialized_(false),
          firstConfigure_(true),
          featureType_(0) {
    addInput(2);
    addOutput(1);
    if (paramCompressed(config))
        cacheWriter_ = new CompressedSortedCacheWriter;
    else
        cacheWriter_ = new SortedCacheWriter;
    cacheWriter_->setCacheDirectory(paramCacheDirectory(config));
    bufferSize_     = paramBufferSize(config);
    repeatFeatures_ = paramRepeatFeatures(config);
    ModelCombination modelCombination(select("model-combination"), ModelCombination::useAcousticModel,
                                      Am::AcousticModel::noEmissions | Am::AcousticModel::noStateTransition);
    modelCombination.load();
    acousticModel_ = modelCombination.acousticModel();
}

AlignedFeatureCacheWriterNode::~AlignedFeatureCacheWriterNode() {
    delete cacheWriter_;
}

void Speech::AlignedFeatureCacheWriterNode::initialize() {
    cacheWriter_->initialize(acousticModel_->nEmissions(), bufferSize_);
    initialized_ = true;
}

bool AlignedFeatureCacheWriterNode::setParameter(const std::string& name, const std::string& value) {
    if (paramCacheDirectory.match(name))
        cacheWriter_->setCacheDirectory(value);
    else if (paramBufferSize.match(name))
        bufferSize_ = paramBufferSize(value);
    else if (paramRepeatFeatures.match(name))
        repeatFeatures_ = paramRepeatFeatures(value);
    else
        return false;
    return true;
}

bool AlignedFeatureCacheWriterNode::configure() {
    Core::Ref<Flow::Attributes> alignmentAttributes(new Flow::Attributes());
    Core::Ref<Flow::Attributes> featureAttributes(new Flow::Attributes());

    getInputAttributes(0, *alignmentAttributes);
    getInputAttributes(1, *featureAttributes);
    if (!configureDatatype(alignmentAttributes, Flow::DataAdaptor<Alignment>::type()))
        return false;

    if (firstConfigure_) {
        std::string datatypeName(featureAttributes->get("datatype"));
        featureType_ = Flow::Registry::instance().getDatatype(datatypeName);
        if (!featureType_) {
            error("unknown input datatype: '%s'", datatypeName.c_str());
            return false;
        }
        log("caching features of type: %s", featureType_->name().c_str());
        firstConfigure_ = false;
    }
    else {
        if (!configureDatatype(featureAttributes, featureType_))
            return false;
    }

    return (putOutputAttributes(0, alignmentAttributes) &&
            putOutputAttributes(1, featureAttributes));
}

bool AlignedFeatureCacheWriterNode::work(Flow::PortId) {
    if (!initialized_)
        initialize();
    Flow::DataPtr<Flow::DataAdaptor<Alignment>> in;
    Flow::DataPtr<Flow::Data>                   feature;
    if (!getData(0, in)) {
        return putData(0, in.get());
    }
    Alignment& alignment = (*in)();
    if (alignment.hasWeights())
        error("Weighted alignments are not supported");
    if (alignment.empty()) {
        warning("empty alignment. segment skipped.");
        while (getData(1, feature))
            putData(1, feature.get());
        return putData(0, in.get());
    }

    TimeframeIndex time = 0;

    std::vector<Alignment::Frame> frames;
    alignment.getFrames(frames);
    for (std::vector<Alignment::Frame>::const_iterator i = frames.begin(); i != frames.end(); ++i, ++time) {
        Flow::DataPtr<Flow::Data> newFeature;
        if (!getData(1, newFeature)) {
            if (!repeatFeatures_ || time == 0) {
                error("cannot fetch feature for timeframe %d", time);
                return false;
            }
        }
        else {
            feature = newFeature;
        }
        Alignment::const_iterator item, itemEnd;
        Core::tie(item, itemEnd) = *i;
        verify_(std::distance(item, itemEnd) == 1);

        Mm::MixtureIndex mixture = acousticModel_->emissionIndex(item->emission);
        if (!cacheWriter_->add(feature, mixture))
            error("cannot add feature to feature cache %d", mixture);
        putData(1, feature.get());
    }
    if (getData(1, feature)) {
        error("feature stream and alignment are not synchronized");
    }
    putData(1, feature.get());
    return putData(0, in.get());
}

// ==========================================

const Core::ParameterString AlignedFeatureCacheReaderNode::paramCacheDirectory(
        "path", "cache directory", ".");
const Core::ParameterBool AlignedFeatureCacheReaderNode::paramCompressed(
        "compressed", "compress caches by counting consecutive equal features", false);
const Core::ParameterInt AlignedFeatureCacheReaderNode::paramBufferSize(
        "buffer-size", "number of feature vectors to buffer", 1024);
const Core::ParameterString AlignedFeatureCacheReaderNode::paramDatatype(
        "datatype", "datatype of the cached objects", Feature::FlowFeature::type()->name().c_str());
const Core::ParameterBool AlignedFeatureCacheReaderNode::paramIgnoreCacheErrors(
        "ignore-cache-errors", "ignore caches that cannot be opened", false);

AlignedFeatureCacheReaderNode::AlignedFeatureCacheReaderNode(const Core::Configuration& config)
        : Core::Component(config),
          Flow::SleeveNode(config),
          currentId_(-1),
          haveLabel_(false),
          cacheDirChanged_(false) {
    if (paramCompressed(config))
        cacheReader_ = new CompressedSortedCacheReader;
    else
        cacheReader_ = new SortedCacheReader;
    cacheReader_->setCacheDirectory(paramCacheDirectory(config));
    cacheReader_->setBufferSize(paramBufferSize(config));
    ignoreCacheErrors_ = paramIgnoreCacheErrors(config);
    if (!cacheReader_->setDatatype(paramDatatype(config)))
        error("cannot create datatype %s", paramDatatype(config).c_str());
}

AlignedFeatureCacheReaderNode::~AlignedFeatureCacheReaderNode() {
    delete cacheReader_;
}

bool AlignedFeatureCacheReaderNode::setParameter(const std::string& name, const std::string& value) {
    if (paramCacheDirectory.match(name)) {
        cacheReader_->setCacheDirectory(value);
        cacheDirChanged_ = true;
    }
    else if (paramBufferSize.match(name)) {
        cacheReader_->setBufferSize(paramBufferSize(value));
    }
    else if (paramIgnoreCacheErrors.match(name)) {
        ignoreCacheErrors_ = paramIgnoreCacheErrors(value);
    }
    else if (paramDatatype.match(name)) {
        if (!cacheReader_->setDatatype(paramDatatype(value))) {
            error("cannot create datatype %s", value.c_str());
            return false;
        }
    }
    else {
        return false;
    }
    return true;
}

bool AlignedFeatureCacheReaderNode::configure() {
    verify(cacheReader_->datatype());
    haveLabel_ = false;
    Core::Ref<Flow::Attributes> labelAttributes(new Flow::Attributes);
    getInputAttributes(0, *labelAttributes);
    if (!configureDatatype(labelAttributes, Flow::String::type()))
        return false;
    Core::Ref<Flow::Attributes> attributes(new Flow::Attributes());
    attributes->set("datatype", cacheReader_->datatype()->name());
    return putOutputAttributes(0, attributes);
}

bool AlignedFeatureCacheReaderNode::work(Flow::PortId) {
    if (!haveLabel_) {
        Flow::DataPtr<Flow::String> in;
        if (!Flow::Node::getData(0, in)) {
            return putEos(0);
        }
        if (!setId(in->data())) {
            return putEos(0);
        }
        haveLabel_ = true;
    }

    Flow::DataPtr<Flow::Data> out = cacheReader_->getData();
    if (!out) {
        haveLabel_ = false;
        return putEos(0);
    }
    return putData(0, out.get());
}

bool AlignedFeatureCacheReaderNode::setId(const std::string& strId) {
    s32  id     = atoi(strId.c_str());
    bool retVal = true;
    if (id != currentId_ || cacheDirChanged_) {
        if (!cacheReader_->open(id)) {
            if (ignoreCacheErrors_)
                warning("failed to open cache for %d. skipped cache.", id);
            else
                error("failed to open cache for %d.", id);
            retVal = false;
        }
    }
    currentId_       = id;
    cacheDirChanged_ = false;
    return retVal;
}
