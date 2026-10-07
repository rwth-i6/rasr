target_sources(
        RasrSearch
                PRIVATE LlmHistoryTrie.cc LlmScorer.cc LlmTimesyncBeamSearch.cc
                        LlmWordScorer.cc WordAssembler.cc)
