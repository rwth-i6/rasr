#ifndef _SPEECH_ALIGNED_FEATURE_CACHE_HH
#define _SPEECH_ALIGNED_FEATURE_CACHE_HH

#include <Am/AcousticModel.hh>
#include "Feature.hh"

namespace Speech {

/**
 * base class of SortedCache reader and writer classes.
 * an aligned feature cache stores feature sorted by mixture ids.
 */
class SortedCache {
protected:
    typedef Flow::DataPtr<Flow::Data> FeaturePtr;
    typedef std::vector<FeaturePtr>   FeatureBuffer;

public:
    SortedCache()
            : cacheDir_("."),
              bufferSize_(0) {}
    SortedCache(const std::string& cacheDir, u32 bufferSize)
            : cacheDir_(cacheDir),
              bufferSize_(bufferSize) {}

    /**
     * Set the directory for feature cache files
     */
    void setCacheDirectory(const std::string& dir) {
        cacheDir_ = dir;
    }

protected:
    std::string cacheFile(u32 id, bool compressed = false) const {
        return Core::form("%s/%d%s", cacheDir_.c_str(), id, (compressed ? ".gz" : ""));
    }

    std::string cacheDir_;
    u32         bufferSize_;
};

class SortedCacheWriter : public SortedCache {
protected:
    typedef SortedCache::FeaturePtr    FeaturePtr;
    typedef SortedCache::FeatureBuffer FeatureBuffer;

public:
    SortedCacheWriter() {}
    SortedCacheWriter(u32 nIds, const std::string& cacheDir, u32 bufferSize);
    virtual ~SortedCacheWriter();

    /**
     * @param nIds       number of different ids used
     * @param bufferSize number of features to cache in memory before
     *                   writing them to disk
     */
    virtual void initialize(u32 nIds, u32 bufferSize = 1024);

    /**
     * Add a new feature to the cache
     */
    virtual bool add(FeaturePtr feature, u32 id);

protected:
    bool         writeCache(u32 id) const;
    virtual void writeBuffer(Core::BinaryOutputStream& out, u32 id) const;
    virtual void clearBuffer(u32 id);
    bool         finish();

    std::vector<FeatureBuffer> featureCaches_;
};

/**
 * implements a simple way of compression by counting
 * equal consecutive features
 */
class CompressedSortedCacheWriter : public SortedCacheWriter {
    typedef std::vector<u8> CountBuffer;

public:
    CompressedSortedCacheWriter() {}
    CompressedSortedCacheWriter(u32 nIds, const std::string& cachedir, u32 bufferSize);
    virtual ~CompressedSortedCacheWriter();

    virtual void initialize(u32 nIds, u32 bufferSize = 1024);
    virtual bool add(FeaturePtr feature, u32 id);

protected:
    virtual void             writeBuffer(Core::BinaryOutputStream& out, u32 id) const;
    virtual void             clearBuffer(u32 id);
    std::vector<CountBuffer> countBuffers_;
};

class SortedCacheReader : public SortedCache {
protected:
    typedef SortedCache::FeaturePtr FeaturePtr;
    typedef std::deque<FeaturePtr>  FeatureBuffer;

public:
    SortedCacheReader()
            : open_(false),
              featureType_(0) {}
    SortedCacheReader(const std::string& cacheDir, u32 bufferSize)
            : SortedCache(cacheDir, bufferSize),
              open_(false),
              featureType_(0) {}
    virtual ~SortedCacheReader() {}

    /**
     * set buffer size
     */
    void setBufferSize(u32 bufferSize) {
        bufferSize_ = bufferSize;
    }
    /**
     * Open cache with id @c id
     */
    bool open(u32 id);
    /**
     * Read feature from cache
     * @return feature or 0 if all features have been read.
     */
    FeaturePtr getData();
    /**
     * set the datatype of the objects to read
     */
    bool                  setDatatype(const std::string& datatype);
    const Flow::Datatype* datatype() const {
        return featureType_;
    }

protected:
    void                    close();
    virtual void            fillBuffer();
    FeatureBuffer           buffer_;
    Core::BinaryInputStream stream_;
    std::streampos          streamEnd_;
    bool                    open_;
    const Flow::Datatype*   featureType_;
};

class CompressedSortedCacheReader : public SortedCacheReader {
public:
    CompressedSortedCacheReader() {}
    CompressedSortedCacheReader(const std::string& cacheDir, u32 bufferSize)
            : SortedCacheReader(cacheDir, bufferSize) {}
    virtual ~CompressedSortedCacheReader() {}

protected:
    virtual void fillBuffer();
};

/**
 * write aligned features to feature caches.
 *
 * parameter @c compressed can be used to compress
 * discrete features, e.g. like speaker labels
 *
 * use parameter @c repeat-features for segment "features"
 * like segment ids or speaker labels
 *
 * parameter size @c buffer-size can be used to control
 * the amount of used memory
 *
 * input: alignment (Speech::Alignment)
 *        feature   (Flow::Data)
 * output = input
 */
class AlignedFeatureCacheWriterNode
        : public Flow::Node {
private:
    static const Core::ParameterString paramCacheDirectory;
    static const Core::ParameterBool   paramCompressed;
    static const Core::ParameterInt    paramBufferSize;
    static const Core::ParameterBool   paramRepeatFeatures;

protected:
    void initialize();

public:
    AlignedFeatureCacheWriterNode(const Core::Configuration&);
    virtual ~AlignedFeatureCacheWriterNode();

    virtual Flow::PortId getInput(const std::string& name) {
        return (name == "features" ? 1 : 0);
    }
    virtual Flow::PortId getOutput(const std::string& name) {
        return (name == "features" ? 1 : 0);
    }

    virtual bool setParameter(const std::string& name, const std::string& value);
    virtual bool configure();
    virtual bool work(Flow::PortId);

    static std::string filterName() {
        return "speech-aligned-feature-cache-writer";
    }

private:
    SortedCacheWriter*           cacheWriter_;
    bool                         initialized_, firstConfigure_;
    bool                         repeatFeatures_;
    Core::Ref<Am::AcousticModel> acousticModel_;
    u32                          bufferSize_;
    const Flow::Datatype*        featureType_;
};

/**
 * read feature from a aligned feature cache
 *
 * parameters @c datatype and @c compressed have
 * have to correspond to the used cache files.
 *
 * parameter @c ignore-cache-errors can be used to
 * tollerate missing caches
 *
 * input:  label (string)
 * output: feature (Flow::TypedAggregate<Flow::Vector<Mm::FeatureType>>)
 */
class AlignedFeatureCacheReaderNode
        : public Flow::SleeveNode {
private:
    static const Core::ParameterString paramCacheDirectory;
    static const Core::ParameterBool   paramCompressed;
    static const Core::ParameterInt    paramBufferSize;
    static const Core::ParameterBool   paramIgnoreCacheErrors;
    static const Core::ParameterString paramDatatype;

public:
    AlignedFeatureCacheReaderNode(const Core::Configuration&);
    virtual ~AlignedFeatureCacheReaderNode();

    virtual bool setParameter(const std::string& name, const std::string& value);
    virtual bool configure();
    virtual bool work(Flow::PortId);

    static std::string filterName() {
        return "speech-aligned-feature-cache-reader";
    }

private:
    SortedCacheReader* cacheReader_;
    bool               setId(const std::string& id);
    s32                currentId_;
    bool               haveLabel_, cacheDirChanged_;
    bool               ignoreCacheErrors_;
};

}  // namespace Speech

#endif  // _SPEECH_ALIGNED_FEATURE_CACHE_HH
