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
#include <unordered_map>

#include <Core/MemoryInfo.hh>
#include <OpenFst/Count.hh>
#include <OpenFst/ReplaceFst.hh>
#include <OpenFst/Scale.hh>
#include <Search/Wfst/GrammarFst.hh>
#include <fst/arcsort.h>
#include <fst/compose.h>
#include <fst/relabel.h>

using namespace Search::Wfst;

AbstractGrammarFst* AbstractGrammarFst::create(GrammarType type, const Core::Configuration& c) {
    AbstractGrammarFst* r = 0;
    switch (type) {
        case TypeVector: r = new GrammarFst(); break;
        case TypeConst: r = new ConstGrammarFst(); break;
        case TypeCompact: r = new CompactGrammarFst(); break;
        case TypeCombine: r = new CombinedGrammarFst(c); break;
        case TypeCompose: r = new ComposedGrammarFst(c); break;
        case TypeDynamic: r = new DynamicGrammarFst(c); break;
        case TypeFailArc: r = new FailArcGrammarFst(); break;
        // case TypeNGram: r = new NGramGrammarFst(); break;
        default: defect(); break;
    }
    return r;
}

void GrammarFst::relabel(const GrammarRelabelerBase& relabeler) {
    relabeler.apply(fst_);
    FstLib::ArcSort(fst_, FstLib::StdILabelCompare());
}

const OpenFst::Label FailArcGrammarFst::FailLabel = -2;

bool FailArcGrammarFst::load(const std::string& filename) {
    typedef std::vector<std::pair<OpenFst::Label, OpenFst::Label>> LabelMapping;
    if (GrammarFst::load(filename)) {
        LabelMapping map(1);
        map.front() = std::make_pair(OpenFst::Epsilon, FailLabel);
        FstLib::Relabel(fst_, map, LabelMapping());
        return true;
    }
    else {
        return false;
    }
}

const Core::ParameterInt CombinedGrammarFst::paramCacheSize(
        "cache", "cache size of the ReplaceFst", 0);
const Core::ParameterStringVector CombinedGrammarFst::paramAddOnFiles(
        "addon-file", "add on fst files", ",");
const Core::ParameterStringVector CombinedGrammarFst::paramReplaceLabels(
        "replace-label", "labels to be replaced with the respective add on fst", ",");
const Core::ParameterIntVector CombinedGrammarFst::paramReplaceIds(
        "replace-id", "label ids to be replaced with the respective add on fst", ",");
const Core::ParameterFloatVector CombinedGrammarFst::paramAddOnScales(
        "addon-scale", "scaling factor applied to the respective add on fst", ",");

CombinedGrammarFst::~CombinedGrammarFst() {
    delete fst_;
    delete rootFst_;
    for (std::vector<OpenFst::VectorFst*>::const_iterator i = addOnFsts_.begin();
         i != addOnFsts_.end(); ++i)
        delete *i;
}

const FstLib::StdFst* CombinedGrammarFst::getFst() const {
    return fst_;
}

bool CombinedGrammarFst::load(const std::string& root) {
    log("loading root fst: %s", root.c_str());
    rootFst_ = OpenFst::VectorFst::Read(root);
    if (!rootFst_) {
        error("error loading %s", root.c_str());
        return false;
    }
    const std::vector<std::string> addOnFiles = paramAddOnFiles(config);
    for (std::vector<std::string>::const_iterator f = addOnFiles.begin(); f != addOnFiles.end(); ++f) {
        log("loading add on fst: %s", f->c_str());
        OpenFst::VectorFst* fst = OpenFst::VectorFst::Read(*f);
        if (!fst) {
            error("error loading %s", f->c_str());
            return false;
        }
        addOnFsts_.push_back(fst);
    }
    const std::vector<std::string> labels = paramReplaceLabels(config);
    if (!labels.empty()) {
        verify(rootFst_->OutputSymbols());
        for (std::vector<std::string>::const_iterator l = labels.begin(); l != labels.end(); ++l) {
            OpenFst::Label id = rootFst_->OutputSymbols()->Find(*l);
            verify(id >= 0);
            replaceLabels_.push_back(id);
            log("using replace label: %s = %d", l->c_str(), id);
        }
    }
    else {
        const std::vector<int> ids = paramReplaceIds(config);
        for (std::vector<int>::const_iterator i = ids.begin(); i != ids.end(); ++i) {
            replaceLabels_.push_back(*i);
            log("using replace label: %d", *i);
        }
    }
    verify(replaceLabels_.size() == addOnFsts_.size());

    const std::vector<f64> scales = paramAddOnScales(config);
    for (u32 i = 0; i < std::min(scales.size(), addOnFsts_.size()); ++i) {
        f32 scale = scales[i];
        log("applying scale to add on fst %d: %f", i, scale);
        if (scale != 1.0) {
            OpenFst::scaleWeights(addOnFsts_[i], scale);
        }
    }
    replaceArcLabels();
    return true;
}

void CombinedGrammarFst::replaceArcLabels() {
    typedef std::pair<OpenFst::Label, OpenFst::Label> LabelPair;
    std::vector<LabelPair>                            iLabelMap, oLabelMap;
    for (u32 i = 0; i < addOnFsts_.size(); ++i) {
        OpenFst::Label uniqLabel = -(i + 1);
        log("add on %i: mapping label %d to %d", i, replaceLabels_[i], uniqLabel);
        oLabelMap.push_back(std::make_pair(replaceLabels_[i], uniqLabel));
        iLabelMap.push_back(std::make_pair(replaceLabels_[i], 0));
    }
    /* HACK: The 1.3.2 version of OpenFst contains an additional check in comparison to 1.2 trunk in
     * src/include/fst/relabel.h that fails for some reason. It looks like commenting that check
     * does not hurt, so why not to do it.
     * [kozielski]
     * */
    log("If you get an error below you need to comment that check in OpenFST (src/include/fst/relabel.h) and link against the new version.");
    FstLib::Relabel(rootFst_, iLabelMap, oLabelMap);
}

void CombinedGrammarFst::relabel(const GrammarRelabelerBase& relabeler) {
    relabeler.apply(rootFst_);
    FstLib::ArcSort(rootFst_, FstLib::StdILabelCompare());
    for (std::vector<OpenFst::VectorFst*>::iterator f = addOnFsts_.begin(); f != addOnFsts_.end(); ++f) {
        relabeler.apply(*f);
        FstLib::ArcSort(*f, FstLib::StdILabelCompare());
    }
    log("relabeled G and add on G");
}

void CombinedGrammarFst::reset() {
    typedef OpenFst::CompactReplaceFst<FstLib::StdArc> ReplaceFst;
    FLAGS_v = 2;
    delete fst_;
    FLAGS_v = 0;
    FstLib::CacheOptions options;
    options.gc_limit = paramCacheSize(config);
    options.gc       = true;

    std::vector<ReplaceFst::PartDefinition> def;
    for (u32 i = 0; i < addOnFsts_.size(); ++i) {
        OpenFst::Label replaceLabel = -(i + 1);
        def.push_back(ReplaceFst::PartDefinition(replaceLabel, addOnFsts_[i]));
    }
    fst_ = new ReplaceFst(rootFst_, def, options);
    log("created ReplaceFst cache=%zd", options.gc_limit);
}

// =======================================

const Core::ParameterInt ComposedGrammarFst::paramCacheSize(
        "cache", "cache size of the ReplaceFst", 0);
const Core::ParameterString ComposedGrammarFst::paramAddOnFile(
        "addon-file", "add on fst", "");
const Core::ParameterFloat ComposedGrammarFst::paramAddOnScale(
        "addon-scale", "scaling factor applied to the add on fst", 1.0);

ComposedGrammarFst::~ComposedGrammarFst() {
    FLAGS_v = 2;
    delete cfst_;
    delete pfst_;
    delete rfst_;
    delete rootFst_;
    delete addOnFst_;
    FLAGS_v = 0;
}

bool ComposedGrammarFst::load(const std::string& root) {
    log("loading root fst: %s", root.c_str());
    rootFst_ = OpenFst::VectorFst::Read(root);
    if (!rootFst_)
        error("error loading %s", root.c_str());
    const std::string addOn = paramAddOnFile(config);
    log("loading add on fst: %s", addOn.c_str());
    addOnFst_ = OpenFst::VectorFst::Read(addOn);
    if (!addOnFst_)
        error("error loading %s", addOn.c_str());
    if (!(rootFst_ && addOnFst_)) {
        return false;
    }
    f32 scale = paramAddOnScale(config);
    if (scale != 1.0) {
        log("applying scale to add on fst: %f", scale);
        OpenFst::scaleWeights(addOnFst_, scale);
    }
    rootFst_->SetOutputSymbols(0);
    addOnFst_->SetInputSymbols(0);
    FstLib::ArcSort(addOnFst_, FstLib::StdILabelCompare());
    return true;
}

void ComposedGrammarFst::relabel(const GrammarRelabelerBase& relabeler) {
    relabeler.getMap(&iLabelMap_);
    log("relabeling map with %zd entries", iLabelMap_.size());
    const OpenFst::SymbolTable* symbols   = rootFst_->InputSymbols();
    OpenFst::Label              freeLabel = symbols->AvailableKey();
    log("using dummy label: %d", freeLabel);
    std::unordered_set<OpenFst::Label> mappedLabels(iLabelMap_.size());
    for (GrammarRelabelerBase::LabelMap::const_iterator i = iLabelMap_.begin();
         i != iLabelMap_.end(); ++i) {
        mappedLabels.insert(i->first);
    }
    for (OpenFst::Label l = 1; l < freeLabel; ++l) {
        if (!mappedLabels.count(l)) {
            iLabelMap_.push_back(std::make_pair(l, freeLabel));
            log("unmapped symbol: %d %s", l, symbols->Find(l).c_str());
        }
    }
    log("updated relabeling map with %zd entries", iLabelMap_.size());
    FstLib::ArcSort(rootFst_, FstLib::StdILabelCompare());
    FstLib::ArcSort(addOnFst_, FstLib::StdILabelCompare());
}

void ComposedGrammarFst::reset() {
    FLAGS_v = 3;
    delete cfst_;
    delete pfst_;
    delete rfst_;
    FLAGS_v = 0;
    FstLib::ComposeFstOptions<FstLib::StdArc, Matcher, Filter> options;
    table_ = options.state_table = new StateTable(*rootFst_, *addOnFst_);
    options.gc_limit             = paramCacheSize(config);
    options.gc                   = true;
    FLAGS_v                      = 2;
    cfst_                        = new ComposeFst(*rootFst_, *addOnFst_, options);
    verify(cfst_);
    pfst_ = new ProjectFst(*cfst_, FstLib::PROJECT_OUTPUT);
    FstLib::RelabelFstOptions relabelOptions;
    relabelOptions.gc_limit = 1024 * 1024;
    rfst_                   = new RelabelFst(*pfst_, iLabelMap_, oLabelMap_, relabelOptions);
    FLAGS_v                 = 0;
    log("created ComposeFst cache=%zd", options.gc_limit);
}

// =======================================

const Core::ParameterBool DynamicGrammarFst::paramLemma(
        "lemma-labels", "use lemma id as labels", true);

const Core::ParameterFloat DynamicGrammarFst::paramPronunciationScale(
        "pronunciation-scale", "scaling of pronunciation scores", 0.0);

bool DynamicGrammarFst::load(const std::string&) {
    require(lexicon_);
    lm_ = Lm::Module::instance().createLanguageModel(select("lm"), lexicon_);
    return lm_;
}

DynamicGrammarFst::~DynamicGrammarFst() {
    delete fst_;
}

void DynamicGrammarFst::relabel(const GrammarRelabelerBase& relabeler) {
    relabeler.getMap(&labelMap_);
}

void DynamicGrammarFst::reset() {
    require(lm_);
    delete fst_;
    DynamicLmFstOptions opts;
    opts.lm                 = lm_;
    opts.outputType         = (paramLemma(config) ? OutputLemma : OutputLemmaPronunciation);
    opts.pronunciationScale = paramPronunciationScale(config);
    opts.gc                 = true;
    opts.gc_limit           = 1024 * 1024 * 100; /*! @todo add parameter */
    fst_                    = new DynamicLmFst(opts);
    if (!labelMap_.empty())
        fst_->SetLabelMapping(labelMap_);
    log("created dynamic lm fst");
    if (opts.outputType == OutputLemmaPronunciation)
        log("using lemma pronunciation output. pronunciation scale=%f", opts.pronunciationScale);
    else
        log("using lemma output");
}
