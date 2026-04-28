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
#ifndef SEARCH_BATCHMANAGER_HH
#define SEARCH_BATCHMANAGER_HH

#include <vector>
#include <assert.h>

#include <Core/Hash.hh>

namespace Tools {
/**
 * For performance-reasons this iterator should be used in a different way than standard iterators.
 * It's underlying Data-Structure is much more flexible than the one used with the old iterators,
 * but when used the right way these iterators provide about the same performance as the old ones.
 * They only allow forward-iteration.
 *
 * usage-example:
 * BatchIndexIterator <...,...> it;
 *
 * for(it = ...; it; ++it)
 * {
 * //use the iterator-value *it
 * }
 *
 * WARNING: Unlike index iterators, pointer-iterators become invalid and may lead to crashes
 * whenever the Array they iterate through is reallocated, so there should be no changes in the
 * Structure for their lifetime.
 */

template<class BatchIdType, class NodeIdType, class NodeType, const BatchIdType invalidBatch_, const BatchIdType singleBatchMask_, const bool debug_ = false>
class BatchPointerIterator {
private:
    std::vector<NodeIdType>& batches_;
    std::vector<NodeType>&   nodes_;
    BatchIdType              batchId_;  // The ID of the current Batch
    NodeType*                currentBorder_;
    NodeType*                current_;

    bool valid_;

    inline NodeIdType getBorderNode() {
        return batches_[batchId_ + 2];
    }

    inline BatchIdType getNextBatch() {
        if (batchId_ & singleBatchMask_)
            return invalidBatch_;

        return (BatchIdType)batches_[batchId_ + 1];
    }

    inline NodeIdType getBatchStartingNode() {
        return batches_[batchId_];
    }

    inline NodeType* getBorderPointer() {
        return &(nodes_[getBorderNode()]);
    }

    inline NodeType* getBatchStartingPointer() {
        return &(nodes_[getBatchStartingNode()]);
    }

public:
    BatchPointerIterator(BatchIdType batchId, std::vector<NodeIdType>& batches, std::vector<NodeType>& values)
            : batches_(batches),
              nodes_(values),
              batchId_(batchId),
              valid_(true) {
        if (batchId_ == invalidBatch_) {
            valid_ = false;
            return;
        };

        if (batchId & singleBatchMask_) {
            current_       = batchId & (~singleBatchMask_);
            currentBorder_ = current_ + 1;
            return;
        }

        currentBorder_ = getBorderPointer();
        current_       = getBatchStartingPointer();
        if (current_ == currentBorder_)
            NextBatch();
    }

    void NextBatch() {
        batchId_ = getNextBatch();
        if (batchId_ == invalidBatch_) {
            current_ = NULL;
            valid_   = false;
        }
        else {
            current_       = getBatchStartingPointer();
            currentBorder_ = getBorderPointer();
            if (current_ == currentBorder_)
                NextBatch();
        }
    }

    bool operator==(const BatchPointerIterator<BatchIdType, NodeIdType, NodeType, invalidBatch_, singleBatchMask_>& rhs) const {
        return current_ == rhs.current_;
    }

    inline BatchPointerIterator<BatchIdType, NodeIdType, NodeType, invalidBatch_, singleBatchMask_>& operator++() {
        if (debug_)
            verify(valid_);
        ++current_;
        if (current_ == currentBorder_) {
            NextBatch();
        }
        return *this;
    }

    inline NodeType* operator*() const {
        if (debug_)
            verify(valid_);
        return current_;
    }

    /** Returns the index of the current position in the array
     */

    inline NodeIdType index() const {
        return (u32(current_) - u32(&(nodes_[0]))) / sizeof(NodeType);
    }

    inline operator bool() const {
        return valid_;
    }

    /**Counts the the steps from the current positionuntil the end of the sequence is reached.
     ***Warning: Since all links have to be followed, this has a bad worst-case-performance. */
    NodeIdType countToEnd() {
        BatchIdType cid   = batchId_;
        NodeIdType  count = currentBorder_ - current_;

        while ((cid = (BatchIdType)batches_[cid + 1]) != invalidBatch_) {
            count += batches_[cid + 2] - batches_[cid];
        }

        return count;
    }
};

template<class BatchIdType, class NodeIdType, const BatchIdType invalidBatch_, const BatchIdType singleBatchMask_, const bool debug_ = false>
class BatchIndexIterator {
private:
    const std::vector<NodeIdType>* batches_;
    BatchIdType                    batchId_;  // The ID of the current Batch
    NodeIdType                     currentBorder_;
    NodeIdType                     current_;
    bool                           valid_;

    inline NodeIdType getBorderNode() const {
        return (*batches_)[batchId_ + 2];
    }
    inline BatchIdType getNextBatch() const {
        return (BatchIdType)(*batches_)[batchId_ + 1];
    }
    inline NodeIdType getBatchStartingNode() const {
        return (*batches_)[batchId_];
    }

public:
    inline BatchIndexIterator(BatchIdType batchId, const std::vector<NodeIdType>& batches)
            : batches_(&batches),
              batchId_(batchId),
              valid_(true) {
        if (batchId & singleBatchMask_) {
            current_       = batchId & (~singleBatchMask_);
            currentBorder_ = current_ + 1;
            return;
        }

        if (batchId_ == invalidBatch_) {
            valid_ = false;
            return;
        }

        currentBorder_ = getBorderNode();
        current_       = getBatchStartingNode();
        if (current_ == currentBorder_)
            NextBatch();
    }

    bool isLastBatch() const {
        if (batchId_ & singleBatchMask_ || batchId_ == invalidBatch_)
            return true;
        return getNextBatch() == invalidBatch_;
    }

    BatchIdType batchId() const {
        return batchId_;
    }

    inline void NextBatch() {
        if (batchId_ & singleBatchMask_) {
            valid_ = false;
            return;
        }

        batchId_ = getNextBatch();
        if (batchId_ == invalidBatch_) {
            valid_ = false;
        }
        else {
            current_       = getBatchStartingNode();
            currentBorder_ = getBorderNode();
            if (current_ == currentBorder_)
                NextBatch();
        }
    }
    bool operator==(const BatchIndexIterator<BatchIdType, NodeIdType, invalidBatch_, singleBatchMask_>& rhs) const {
        return current_ == rhs.current_;
    }
    inline BatchIndexIterator<BatchIdType, NodeIdType, invalidBatch_, singleBatchMask_>& operator++() {
        if (debug_)
            verify(valid_);
        ++current_;
        if (current_ == currentBorder_) {
            NextBatch();
        }
        return *this;
    }

    BatchIndexIterator<BatchIdType, NodeIdType, invalidBatch_, singleBatchMask_>& operator+=(u32 steps) {
        while (steps) {
            if (debug_)
                verify(valid_);
            u32 currentStep = steps;
            if (currentBorder_ - current_ < currentStep)
                currentStep = currentBorder_ - current_;
            steps -= currentStep;
            current_ += currentStep;
            if (current_ == currentBorder_) {
                NextBatch();
            }
        }
        return *this;
    }

    NodeIdType operator*() const {
        if (debug_)
            verify(valid_);
        return current_;
    }

    inline operator bool() const {
        return valid_;
    }

    inline bool ready() const {
        return !valid_;
    }

    /**Counts the the steps from the current position until the end of the sequence is reached.
     ***Warning: Since all links have to be followed, this has a bad worst-case-performance. */
    NodeIdType countToEnd() {
        if (!valid_)
            return 0;
        BatchIdType cid   = batchId_;
        NodeIdType  count = currentBorder_ - current_;

        while ((cid & singleBatchMask_) == 0 && (cid = (BatchIdType)(*batches_)[cid + 1]) != invalidBatch_) {
            count += (*batches_)[cid + 2] - (*batches_)[cid];
        }

        return count;
    }

    /// Returns the count of steps needed until the given node is reached. The node must be part of this batch.
    u32 countUntil(NodeIdType until) {
        verify(until >= current_);

        if (until < currentBorder_)
            return until - current_;  // Standard case

        BatchIdType cid = batchId_;

        NodeIdType count = currentBorder_ - current_;

        while ((cid = (BatchIdType)(*batches_)[cid + 1]) != invalidBatch_) {
            if (until >= (*batches_)[cid] && until < (*batches_)[cid + 2])
                return count + until - (*batches_)[cid];  // the node is contained by the current batch

            count += (*batches_)[cid + 2] - (*batches_)[cid];
        }
        verify(0);  // Should not happen, as it must be in the batch

        return count;
    }
};

template<class BatchIdType, class NodeIdType, class NodeType, const BatchIdType invalidBatch_, const BatchIdType singleBatchMask_>
class BatchBuilder {
public:
    typedef BatchIndexIterator<BatchIdType, NodeIdType, invalidBatch_, singleBatchMask_>             Iterator;
    typedef BatchPointerIterator<BatchIdType, NodeIdType, NodeType, invalidBatch_, singleBatchMask_> PointerIterator;
    typedef BatchBuilder<BatchIdType, NodeIdType, NodeType, invalidBatch_, singleBatchMask_>         Self;
    typedef Core::HashMap<const NodeIdType, NodeIdType>                                              NodeMap;
    typedef Core::HashMap<const BatchIdType, BatchIdType>                                            BatchMap;

    static const int batchSize = 3;

    std::vector<NodeIdType>& batches_;  /// the batches
    std::vector<NodeType>&   nodes_;

    inline void setBatchFollower(const BatchIdType batch, const BatchIdType follower) {
        verify(!(batch & singleBatchMask_));
        batches_[batch + 1] = follower;
    }

    inline BatchIdType getBatchFollower(const BatchIdType batch) {
        verify(!(batch & singleBatchMask_));
        return batches_[batch + 1];
    }

    inline NodeIdType getBatchStart(const BatchIdType batch) {
        verify(!(batch & singleBatchMask_));
        return batches_[batch];
    }

    inline NodeIdType getBatchEnd(const BatchIdType batch) {
        verify(!(batch & singleBatchMask_));
        return batches_[batch + 2];
    }

    inline bool isBatchEmpty(const BatchIdType batch) {
        verify(!(batch & singleBatchMask_));
        return batches_[batch + 1] == invalidBatch_ && batches_[batch] == batches_[batch + 2];
    }

    inline void emptyBatch(const BatchIdType batch) {
        verify(!(batch & singleBatchMask_));
        batches_[batch]     = 0;
        batches_[batch + 2] = 0;
    }

    inline void clearBatch(const BatchIdType batch) {
        verify(!(batch & singleBatchMask_));
        batches_[batch]     = 0;
        batches_[batch + 1] = invalidBatch_;
        batches_[batch + 2] = 0;
    }

    inline BatchIdType createBatchBehind(const BatchIdType batch, NodeIdType start, NodeIdType end) {
        verify(!(batch & singleBatchMask_));
        int sz = batches_.size();
        batches_.resize(sz + 3);
        batches_[sz]        = start;
        batches_[sz + 2]    = end;
        batches_[sz + 1]    = batches_[batch + 1];
        batches_[batch + 1] = sz;
        return sz;
    }

    inline BatchIdType getLastBatch(BatchIdType batch) {
        verify(!(batch & singleBatchMask_));
        BatchIdType b;
        while ((b = batches_[batch + 1]) != invalidBatch_)
            batch = b;
        return batch;
    }

    /**Appends "slave" directly to "parent" */
    inline void appendToBatchPrivate(BatchIdType parent, BatchIdType slave) {
        verify(!(parent & singleBatchMask_));
        batches_[parent + 1] = slave;
    }

    static void initializeVectors(std::vector<NodeType>& values, std::vector<NodeIdType>& batches) {
        values.clear();
        batches.clear();
    }

    inline void verifyBatch(BatchIdType batch) const {
        verify(!(batch & singleBatchMask_));

        verify(batches_[batch] <= batches_[batch + 2]);
        verify(batches_[batch + 1] != batch);
    }

    inline bool checkBatch(const BatchIdType batch) const {
        verify(!(batch & singleBatchMask_));
        if (batches_[batch] <= batches_[batch + 2] && batches_[batch + 1] != batch)
            return true;
        return false;
    }

    void verifyBatchesChain(BatchIdType batch) const {
        verify(!(batch & singleBatchMask_));
        if (!(batch >= 0 && batch < batches_.size() - 2))
            Core::Application::us()->log() << "error-batch: " << batch << " batches.size: " << batches_.size() << "\n";
        verify(batch >= 0 && batch < batches_.size() - 2);
        while (batch != invalidBatch_) {
            verifyBatch(batch);
            batch = batches_[batch + 1];
        }
    }

    inline bool checkBatchesChain(BatchIdType batch) const {
        verify(!(batch & singleBatchMask_));
        if (batch >= 0 && batch < batches_.size() - 2) {
            batch = batches_[batch + 1];
            if (batch != invalidBatch_) {
                return checkBatchesChain(batch);
            }
            else
                return true;
        }
        else
            return false;
    }

    BatchBuilder(std::vector<NodeIdType>& batches, std::vector<NodeType>& nodes)
            : batches_(batches),
              nodes_(nodes) {
    }

    ~BatchBuilder() {}

    /** Deletes all batches and values, and initializes the structure  */
    void clearInitializeStructure() {
        Self::initializeVectors(nodes_, batches_);
    }

    /**  Returns the index of a value-pointer within the nodes_-vector */
    inline NodeIdType idOfNode(NodeType* val) const {
        return ((u32)((char*)val - (char*)&(nodes_.front()))) / sizeof(NodeType);
    }

    /** The same things about the BatchId as for prepend and append.
     *** These functions should not be used too much, since it is much more efficient to append/prepend as many
     *** values as possible at a time. Returns the dex of the value.
     ***
     ***
     *** generally appending never changes the given BatchId except that its value is invalidBatch_, while
     *** prepending may always change it.
     */

    void print() const {
        for (int a = 0; a < batches_.size(); a++) {
            Core::Application::us()->log() << "Batch " << a << ": " << batches_[a] << ":\n";
        }
    }

    /**
      usage-example:
     ...::Iterator it;

     for(it = ...; it; ++it) {
      //use the iterator-value(which is of the template-type BatchIdType)
     }
    */

    inline Iterator getIterator(BatchIdType batch) const {
        return Iterator(batch, batches_);
    }

    /** This iterator iterates directly through the nodes without the step through indices. That saves a few cpu-cycles.
      usage-example:
      ...::iterator it;

      for(it = ...; it; ++it)  {
       //use the pointer *it
      }
    */

    inline PointerIterator getPointerIterator(BatchIdType batch) const {
        return PointerIterator(batch, batches_, nodes_);
    }
};

/**
 * invalidBatch_ is the index that is interpreted as "no batch".
 * If mergeBatches_ is true, space is saved by using the end-index of one batch as the start-index of the next.
 * In an ideal structure this saves about 1/3 of the batches-arrays size, but with mergeBatches_ batches can only grow,
 * not shrink.
 *
 * Possible optimizations:
 * The batch-chains have no pointer to their end, so the append-operation has to walk through all batches whenever a new batch is appended.
 * For cases where the order doesn't count, it might be useful to add another operation that inserts a new batch into the chain at the second
 * position(behind the first, so the ID of the first does not have to be changed, but without the need to walk through the whole chain)
 *
 * Explanation of the data-structure:
 * Generally this can be used to represent sets of Values(of type NodeType) that belong together(batches), that are stored compactly and can be
 * iterated efficiently.
 */

///@param BatchIdType Type of batch indices (usually u32)
///@param NodeIdType Type of node indices (usually u32)
///@param NodeType Type of the contained nodes.
///@param manageNodes Whether this manager should also manage the node vector.
///                                      If this is true, functions that change the node vector are allowed to be called.
///                                      If it is false, only functions that change the batches are allowed to be called.
///@param invalidBatch_ Value of the "invalid" batch. Usually a special value like Core::Type<BatchIdType>::max()
///@param singleBatchMask_ An optional mask to be used for marking "single" batches containing only one single item.
///                                           The batches batch-id will be directly the id of the node, marked with this mask. This saves
///                                           a lot of space when there is many batches consisting of only one item.
///@param mergeBatches_  Whether batches should be merged. If this is true space is saved by using the end-index
///                                         of one batch as the start-index of the next. In an ideal structure this saves about 1/3
///                                         of the batches-arrays size, but then batches can only grow, not shrink.
template<class BatchIdType, class NodeIdType, class NodeType,
         bool manageNodes, const BatchIdType invalidBatch_, const BatchIdType singleBatchMask_ = 0, const bool mergeBatches_ = true, const bool debug_ = false>
class BatchManager : private BatchBuilder<BatchIdType, NodeIdType, NodeType, invalidBatch_, singleBatchMask_> {
public:
    typedef BatchManager<BatchIdType, NodeIdType, NodeType, invalidBatch_, mergeBatches_, debug_> mytype;
    typedef mytype                                                                                Self;
    typedef BatchBuilder<BatchIdType, NodeIdType, NodeType, invalidBatch_, singleBatchMask_>      SelfBuilder;
    typedef typename SelfBuilder::Iterator                                                        Iterator;
    typedef typename SelfBuilder::PointerIterator                                                 PointerIterator;
    typedef Iterator                                                                              iterator;

private:
    inline BatchIdType createNewBatch(NodeIdType from, NodeIdType to, BatchIdType append = invalidBatch_, BatchIdType appendTo = invalidBatch_, bool forceNormalBatch = false) {
        if (singleBatchMask_ && !forceNormalBatch) {
            if (appendTo == invalidBatch_ && append == invalidBatch_) {
                // Eventually Create a single-batch
                if (to == from + 1)
                    return ((BatchIdType)from) | singleBatchMask_;
            }
        }

        BatchIdType appendToLast = invalidBatch_;
        if (appendTo != invalidBatch_)
            appendToLast = this->getLastBatch(appendTo);

        BatchIdType ret = (BatchIdType)SelfBuilder::batches_.size();

        if (appendTo != invalidBatch_ && this->SelfBuilder::batches_[appendToLast + 2] == from) {
            // The last value of the previous batch is the one before the first value of this batch(checked above),
            // and the last batch is the one that this one should be appendet to.
            // We can simply expand the previous batch.
            SelfBuilder::batches_[appendToLast + 2] = to;
            return appendTo;
        }

        if (!mergeBatches_ || ret == 0 || SelfBuilder::batches_[ret - 1] != from) {
            SelfBuilder::batches_.push_back(from);
        }
        else {
            --ret;  // The Node of "from" can be found at SelfBuilder::batches_[--ret]
        }

        SelfBuilder::batches_.push_back(append);
        SelfBuilder::batches_.push_back(to);

        if (debug_)
            verify(append != ret);

        if (appendTo != invalidBatch_) {
            SelfBuilder::appendToBatchPrivate(appendToLast, ret);
            return appendTo;
        }

        return ret;
    }

public:
    typedef typename SelfBuilder::NodeMap  NodeMap;
    typedef typename SelfBuilder::BatchMap BatchMap;

    using SelfBuilder::verifyBatch;

    inline bool checkBatch(const BatchIdType batch) const {
        return SelfBuilder::checkBatch(batch);
    }

    inline void verifyBatchesChain(BatchIdType batch) const {
        SelfBuilder::verifyBatchesChain(batch);
    }

    inline bool checkBatchesChain(BatchIdType batch) const {
        return SelfBuilder::checkBatchesChain(batch);
    }

    BatchManager(std::vector<NodeIdType>& batches, std::vector<NodeType>& nodes)
            : SelfBuilder(batches, nodes) {
    }

    ~BatchManager() {
    }

    /** Deletes all batches and values, and initializes the structure
     */
    void clearInitializeStructure() {
        SelfBuilder::clearInitializeStructure();
    }

    /**
     *  Returns the index of a value-pointer within the SelfBuilder::nodes_-vector
     */

    inline NodeIdType idOfNode(NodeType* val) const {
        return ((u32)((char*)val - (char*)&(SelfBuilder::nodes_.front()))) / sizeof(NodeType);
    }

    /** The same things about the BatchId as for prepend and append.
     *** These functions should not be used too much, since it is much more efficient to append/prepend as many
     *** values as possible at a time. Returns the dex of the value.
     ***
     ***
     *** generally appending never changes the given inBatchId except that its value is invalidBatch_, while
     *** prepending may always change it.
     */

    NodeIdType prependOne(BatchIdType& id, const NodeType& val) {
        verify(manageNodes);
        NodeType* v = prepend(id, 1);
        *v          = val;

        return idOfNode(v);
    }

    NodeIdType appendOne(BatchIdType& id, const NodeType& val) {
        verify(manageNodes);
        NodeType* v = append(id, 1);
        *v          = val;

        return idOfNode(v);
    }

    /// @warning Only when this manager also manages the nodes
    /// Prepend operations are faster then append operations, since they don't require the chain to be followed.
    /// Prepends @param count new nodes to the batch with id @param id
    /// Changes @p id so contain the new data
    NodeType* prepend(BatchIdType& id, NodeIdType count = 1) {
        verify(manageNodes);
        verify(count == 1);  // currently only 1 is supported

        NodeIdType ret = SelfBuilder::nodes_.size();
        SelfBuilder::nodes_.resize(ret + count);
        prependToBatchPrivate(id, ret, ret + count);
        return &(SelfBuilder::nodes_[ret]);
    }

    /// @warning Only when this manager also manages the nodes
    /// Appends @param count new nodes to the batch with id @param id
    /// Changes @p id so contain the new data
    NodeType* append(BatchIdType& id, NodeIdType count = 1) {
        verify(manageNodes);
        verify(count == 1);  // currently only 1 is supported
        NodeIdType ret = SelfBuilder::nodes_.size();
        SelfBuilder::nodes_.resize(ret + count);
        appendToBatchPrivate(id, ret, ret + count);
        return &(SelfBuilder::nodes_[ret]);
    }

    /// @warning Only when this manager does not manage the nodes, or if history-recording is not used!
    /// Prepends the nodes in the range from @p from to @p to to the batch with id @p id.
    /// Changes @p id to reflect the new content.
    ///
    /// Prepending is faster then appending, since it doesn't require the full batch chain to be walked.
    ///
    /// Append/prepend-operations should always be used with ranges as big as possible, for performance-reasons.
    /// With only ranges of 1(and only prepending), the structure would become a linked list. If the given ID is invalidBatch_, a new batch is created.
    /// Prepend operations can change the id of the batch. The new id is stored into the given reference.
    inline void prependToBatch(BatchIdType& id, NodeIdType from, NodeIdType to) {
        verify(!manageNodes);

        prependToBatchPrivate(id, from, to);
    }

    /// @warning Only when this manager does not manage the nodes, or if history-recording is not used!
    /// Appends the nodes in the range from @p from to @p to to the batch with id @p id.
    /// Changes @p id to reflect the new content.
    ///
    ///  Append/prepend-operations should always be used with ranges as big as possible, for performance-reasons.
    ///  With only ranges of 1(and only prepending), the structure would become a linked list. If the given ID is invalidBatch_, a new batch is created.
    ///  Prepend operations can change the id of the batch. The new id is stored into the given reference.
    inline void appendToBatch(BatchIdType& id, NodeIdType from, NodeIdType to) {
        verify(!manageNodes);
        appendToBatchPrivate(id, from, to);
    }

    inline void print() const {
        SelfBuilder::print();
    };

    /**
      usage-example:
      ...::iterator it;

      for(it = ...; it; ++it) {
      //use the iterator-value(which is of the template-type BatchIdType)
      }
    */

    inline Iterator getIterator(BatchIdType batch) const {
        return SelfBuilder::getIterator(batch);
    }

    /** This iterator iterates directly through the values without the step through indices. That saves a few cpu-cycles.

     usage-example:
     ...::iterator it;

     for(it = ...; it; ++it)  {
      //use the pointer *it
     }
    */

    inline PointerIterator getPointerIterator(BatchIdType batch) const {
        return SelfBuilder::getPointerIterator(batch);
    }

private:
    inline void prependToBatchPrivate(BatchIdType& id, NodeIdType from, NodeIdType to) {
        if (id & singleBatchMask_) {
            // First create a normal batch out of the appended single batch, and then do stuff on it.
            NodeIdType node = id & (~singleBatchMask_);
            id              = createNewBatch(node, node + 1, invalidBatch_, invalidBatch_, true);
        }

        id = createNewBatch(from, to, id);
    }

    inline void appendToBatchPrivate(BatchIdType& id, NodeIdType from, NodeIdType to) {
        if (id == invalidBatch_) {
            prependToBatchPrivate(id, from, to);
            return;
        }

        if (id & singleBatchMask_) {
            // First create a normal batch out of the single batch, and then do stuff on it.
            NodeIdType node = id & (~singleBatchMask_);
            id              = createNewBatch(node, node + 1, invalidBatch_, invalidBatch_, true);
        }

        createNewBatch(from, to, invalidBatch_, id);
    }
};
}  // namespace Tools

#endif  // SEARCH_BATCHMANAGER_HH
