#ifndef _LM_SIMPLE_HISTORY_LM_HH
#define _LM_SIMPLE_HISTORY_LM_HH

#include "LanguageModel.hh"
#include "NNHistoryManager.hh"

namespace Lm {

  /**
   * A simple language model for token history extension and hashing (recombination)
   * also useful for no-LM recognition but still with recombination capability
   */

  struct SimpleHistory
  {
    TokenIdSequence tokIdSeq;
    mutable u32 refCount;

    SimpleHistory(): refCount(0) {}
    SimpleHistory(Bliss::Token::Id tid): tokIdSeq(1, tid), refCount(0) {}
    SimpleHistory(const TokenIdSequence& r, Bliss::Token::Id tid): tokIdSeq(r), refCount(0) { tokIdSeq.push_back(tid); }
  };

  class SimpleHistoryManager : public HistoryManager
  {
    public:
      SimpleHistoryManager() {}
      ~SimpleHistoryManager() {}

      HistoryHandle acquire (HistoryHandle handle) 
      {
        const SimpleHistory* sh = static_cast<const SimpleHistory*>(handle);
        ++(sh->refCount);
        return handle;
      }
    
      virtual void release (HistoryHandle handle)
      {
        const SimpleHistory* sh = static_cast<const SimpleHistory*>(handle);
        --(sh->refCount);
        if ( sh->refCount == 0 )
          delete sh;
      }
  
      HistoryHash hashKey (HistoryHandle handle) const
      {
        const SimpleHistory* sh = static_cast<const SimpleHistory*>(handle);
        return token_id_sequence_hash(sh->tokIdSeq);
      }

      bool isEquivalent(HistoryHandle lhd, HistoryHandle rhd) const
      { // lhd != rhd when reaching here
        const SimpleHistory* lsh = static_cast<const SimpleHistory*>(lhd);
        const SimpleHistory* rsh = static_cast<const SimpleHistory*>(rhd);
        return lsh->tokIdSeq == rsh->tokIdSeq;
      }
  };

  class SimpleHistoryLm : public LanguageModel
  {
      typedef LanguageModel Precursor;
    public:
      SimpleHistoryLm(const Core::Configuration& c, Bliss::LexiconRef l) : Core::Component(c), Precursor(c, l) 
      { historyManager_ = new SimpleHistoryManager(); }

      virtual ~SimpleHistoryLm() { delete historyManager_; }

      // language model interface
      History startHistory() const
      {
        SimpleHistory* sh = new SimpleHistory(sentenceBeginToken()->id());
        return history(sh);
      }

      History extendedHistory(const History& h, Token w) const
      {
        const SimpleHistory* sh = static_cast<const SimpleHistory*>(h.handle());
        SimpleHistory* nsh = new SimpleHistory(sh->tokIdSeq, w->id());
        return history(nsh);
      }

      // reduced history for limited context
      History reducedHistory(const History& h, u32 limit) const 
      { 
        const SimpleHistory* sh = static_cast<const SimpleHistory*>(h.handle());
        if ( limit >= sh->tokIdSeq.size() )
          return h;
        else {
          SimpleHistory* nsh = new SimpleHistory();
          nsh->tokIdSeq.insert(nsh->tokIdSeq.end(), sh->tokIdSeq.end()-limit, sh->tokIdSeq.end());
          return history(nsh);
        }
      }

      // can be used for noLM recognition
      Score score(const History&, Token w) const { return 0.0; }

      std::string formatHistory(const History &h) const
      {
        const SimpleHistory* sh = static_cast<const SimpleHistory*>(h.handle());
        std::string result;
        for (u32 idx=0; idx<sh->tokIdSeq.size(); ++idx)
          result += " " + std::to_string(sh->tokIdSeq.at(idx));
        return result;
      }
  };

} // namespace Lm

#endif // LM_SIMPLE_HISTORY_LM_HH
