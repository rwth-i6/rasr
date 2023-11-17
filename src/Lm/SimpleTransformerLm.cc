#include "SimpleTransformerLm.hh"

namespace Lm {

const Core::ParameterBool paramTransformOuputLog(
  "transform-output-log",    
  "apply log to tensorflow output",       
  false);

const Core::ParameterBool paramTransformOuputNegate(
  "transform-output-negate", 
  "negate tensorflow output (after log)", 
  false);

const Core::ParameterInt paramMaxBatchSize(
  "max-batch-size",
  "maximum number of histories forwarded in one go",
  64, 1);

SimpleTransformerLm::SimpleTransformerLm(const Core::Configuration& c, Bliss::LexiconRef l) : 
  Core::Component(c), Precursor(c, l),
  session_(select("session")),
  loader_(Tensorflow::Module::instance().createGraphLoader(select("loader"))),
  graph_(loader_->load_graph()), // tf::GraphDef, libraries and necessary param names
  max_batch_size_(paramMaxBatchSize(config))
{
  bool transform_output_log = paramTransformOuputLog(config);
  bool transform_output_negate = paramTransformOuputNegate(config);
  if ( transform_output_log and transform_output_negate ) {
    output_transform_function_ = [](Score v){ return -std::log(v); };
    Core::Application::us()->log() << "apply -log(.) to model output";
  } else if ( transform_output_log ) {
    output_transform_function_ = [](Score v){ return std::log(v); };
    Core::Application::us()->log() << "apply log(.) to model output";
  } else if ( transform_output_negate ) {
    output_transform_function_ = [](Score v){ return -v; };
    Core::Application::us()->log() << "apply -(.) to model output";
  } 
}

SimpleTransformerLm::~SimpleTransformerLm()
{
  startHistory_ = History();
  delete historyManager_;
}

// initialization: vocabulary, model graph and start history
void SimpleTransformerLm::load() 
{
  loadVocabulary();
  // create tf::Session with graph(tf::GraphDef) and default initialization of variables 
  session_.addGraph(*graph_);
  // restore model checkpoint
  loader_->initialize(session_);

  // hard-coded IO names
  Tensorflow::TensorInputMap input_map(select("input-map"));
  input_tensor_name = input_map.get_info("word").tensor_name();
  input_length_tensor_name = input_map.get_info("word").seq_length_tensor_name();

  Tensorflow::TensorOutputMap output_map(select("output-map"));
  output_tensor_names_.push_back(output_map.get_info("softmax").tensor_name());

  // no state_vars to be handled in this simple version
  // Note: model graph should always have the default initial state for each run

  // use SimpleScoreHistoryManager for simplicity and flexibility
  delete historyManager_;
  historyManager_ = new SimpleScoreHistoryManager();
  startHistory_ = startHistory();
  // TODO compute the scores at init already ?
}

History SimpleTransformerLm::startHistory() const
{
  if ( startHistory_.isValid() )
    return startHistory_;
  // once only
  Bliss::Token::Id wId = lexicon_mapping_.at(sentenceBeginToken()->id());
  verify( wId < num_outputs_ );
  SimpleScoreHistoryManager* hm = static_cast<SimpleScoreHistoryManager*>(historyManager_);
  HistoryDescriptor* nhd = new HistoryDescriptor(wId);
  std::pair<SimpleHistoryCache::iterator, bool> result = hm->updateCache(nhd);
  verify( result.second ); // must be the only one
  cacheHashQueue_.push_back(nhd->cacheHash);
  return history(nhd);
}

History SimpleTransformerLm::extendedHistory(const History& h, Token w) const
{
  Bliss::Token::Id wId = lexicon_mapping_.at(w->id());
  verify( wId < num_outputs_ );
  SimpleScoreHistoryManager* hm = static_cast<SimpleScoreHistoryManager*>(historyManager_);
  const HistoryDescriptor* chd = static_cast<const HistoryDescriptor*>(h.handle());
  HistoryDescriptor* nhd = new HistoryDescriptor(chd->tokIdSeq, wId);
  
  std::pair<SimpleHistoryCache::iterator, bool> result = hm->updateCache(nhd); 
  if ( result.second ) { // new one
    cacheHashQueue_.push_back(nhd->cacheHash);
  } else { // use the existing one
    delete nhd;
    nhd = result.first->second;
  }
  return history(nhd);
}

Score SimpleTransformerLm::score(const History& h, Token w) const
{
  size_t wId = lexicon_mapping_.at(w->id());
  verify( wId < num_outputs_ );
  const HistoryDescriptor* chd = static_cast<const HistoryDescriptor*>(h.handle());
  if ( !chd->scores.empty() )
    return chd->scores[wId];

  HistoryDescriptor* hd = const_cast<HistoryDescriptor*>(chd);
  makeBatch(hd);
  verify( batch_.size() > 0 && max_batch_len_ > 0 );
  scoreBatch();
  batch_.clear(); 
  max_batch_len_ = 0;

  verify( hd->scores.size() >= num_outputs_ );
  return hd->scores[wId];
}

void SimpleTransformerLm::makeBatch(HistoryDescriptor* hd) const
{ // sort by length ? general search behavior ensures similar length in the ordered queue
  // maybe more important is the score caching to avoid redundant computaton due to pruning
  batch_.push_back(hd);
  max_batch_len_ = hd->tokIdSeq.size();

  const SimpleHistoryCache& cache = static_cast<SimpleScoreHistoryManager*>(historyManager_)->getCache();
  while ( batch_.size() < max_batch_size_ && !cacheHashQueue_.empty() ) {
    size_t hash = cacheHashQueue_.front();
    cacheHashQueue_.pop_front();
    if ( cache.count(hash) == 0 || hash == hd->cacheHash )
      continue;
    HistoryDescriptor* bhd = cache.at(hash);
    if ( !bhd->scores.empty() )
      continue;
    batch_.push_back(bhd);
    if ( bhd->tokIdSeq.size() > max_batch_len_ )
      max_batch_len_ = bhd->tokIdSeq.size();
  }
}

void SimpleTransformerLm::scoreBatch() const
{ // default initializer always 0 ?
  Math::FastMatrix<s32> tokMat(batch_.size(), max_batch_len_);
  Math::FastVector<s32> lenVec(batch_.size());
  for (u32 bIdx = 0; bIdx < batch_.size(); ++bIdx) {
    const TokenIdSequence& tokSeq = batch_[bIdx]->tokIdSeq;
    verify( tokSeq.size() <= max_batch_len_ );
    lenVec[bIdx] = tokSeq.size();
    for (u32 tIdx = 0; tIdx < tokSeq.size(); ++tIdx)
      tokMat.at(bIdx, tIdx) = tokSeq[tIdx];
    for (u32 tIdx = tokSeq.size(); tIdx < max_batch_len_; ++tIdx)
      tokMat.at(bIdx, tIdx) = 0;
  }

  BatchInput inputs;
  BatchOutput outputs;
  inputs.emplace_back(std::make_pair(input_tensor_name, Tensorflow::Tensor::create(tokMat)));
  inputs.emplace_back(std::make_pair(input_length_tensor_name, Tensorflow::Tensor::create(lenVec)));
  // read tensor values should trigger the computation automatically (no state_vars to be updated)
  session_.run(inputs, output_tensor_names_, {}, outputs);

  // process scores: expect always only the last output position (B,V)
  verify(outputs.size() == 1);
  for (u32 bIdx = 0; bIdx < batch_.size(); ++bIdx) {
    std::vector<Score>& scores = batch_[bIdx]->scores;
    outputs[0].get(bIdx, scores);
    if ( output_transform_function_ )
      std::transform(scores.begin(), scores.end(), scores.begin(), output_transform_function_);
  }
}

} // namespace
