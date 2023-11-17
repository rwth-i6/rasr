#ifndef _LM_SIMPLE_TRANSFORMER_LM_HH
#define _LM_SIMPLE_TRANSFORMER_LM_HH

#include <deque>

#include <Tensorflow/GraphLoader.hh>
#include <Tensorflow/Module.hh>
#include <Tensorflow/Session.hh>
#include <Tensorflow/TensorMap.hh>

#include "AbstractNNLanguageModel.hh"
#include "SimpleHistoryLm.hh"

namespace Lm {

struct SimpleScoreHistory: public SimpleHistory
{ // tokSeq and refCount in base
  std::vector<Score> scores;
  size_t cacheHash;

  typedef SimpleHistory Precursor;
  SimpleScoreHistory(Bliss::Token::Id tid): Precursor(tid), cacheHash(0) {}
  SimpleScoreHistory(const TokenIdSequence& r, Bliss::Token::Id tid): Precursor(r, tid), cacheHash(0) {}
};

typedef std::unordered_map<size_t, SimpleScoreHistory*> SimpleHistoryCache;

class SimpleScoreHistoryManager : public SimpleHistoryManager
{
  protected:
    SimpleHistoryCache historyCache_;

  public:
    SimpleScoreHistoryManager() {}
    ~SimpleScoreHistoryManager() {
      for (SimpleHistoryCache::iterator iter=historyCache_.begin(); iter!=historyCache_.end(); ++iter)
        delete (iter->second);
    }

    void release (HistoryHandle handle) {
      const SimpleScoreHistory* sh = static_cast<const SimpleScoreHistory*>(handle);
      --(sh->refCount); // mutable
      if ( sh->refCount == 0 ) {
        historyCache_.erase(sh->cacheHash);
        delete sh;
      }
    }

    const SimpleHistoryCache& getCache() const { return historyCache_; }

    std::pair<SimpleHistoryCache::iterator, bool> updateCache(SimpleScoreHistory* sh) {
      sh->cacheHash = token_id_sequence_hash(sh->tokIdSeq);
      return historyCache_.insert(std::make_pair(sh->cacheHash, sh));
    }
};

typedef std::vector<std::pair<std::string, Tensorflow::Tensor>> BatchInput;
typedef std::vector<Tensorflow::Tensor> BatchOutput;

// simple TF Transformer LM: mainly for E2E systems with small search space
// trade speed for simplicity: always feed-in full sequence and get last output scores
// Note: slice last position should be done in model graph
class SimpleTransformerLm: public AbstractNNLanguageModel
{
    typedef AbstractNNLanguageModel Precursor;
    typedef SimpleScoreHistory HistoryDescriptor;

  protected:
    // Note: graph related params follow python naming scheme
    mutable Tensorflow::Session              session_;
    std::unique_ptr<Tensorflow::GraphLoader> loader_;
    std::unique_ptr<Tensorflow::Graph>       graph_;

    // should be single input/output tensor
    std::string input_tensor_name;
    std::string input_length_tensor_name; 
    std::vector<std::string> output_tensor_names_;

  protected:
    std::function<Score(Score)> output_transform_function_;
    u32 max_batch_size_; // B
    mutable u32 max_batch_len_;  // T
    mutable std::deque<size_t> cacheHashQueue_; // only not-scored history
    mutable std::vector<HistoryDescriptor*> batch_;

    History startHistory_; // always cached: same scoring

  protected:
    void load();

    // actually no const functions at all for NNLM: just legacy to LM interface
    void makeBatch(HistoryDescriptor* hd) const;
    void scoreBatch() const;

    // cache most recent scored histories to avoid redundant computation due to pruning
    // this can be done by the lookahead table caching scheme (just need to hold the history)
    // but better reduce cache size for memory

  public:
    SimpleTransformerLm(const Core::Configuration& c, Bliss::LexiconRef l);
    ~SimpleTransformerLm();

    // history (no reduction)
    History startHistory() const;
    History extendedHistory(const History& h, Token w) const;
   
    // scoring
    Score score(const History& h, Token w) const;
};

} // namespace Lm

#endif // _LM_SIMPLE_TRANSFORMER_LM_HH
