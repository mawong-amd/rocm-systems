/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hip_graph_internal.hpp"

#define CASE_STRING(X, C)                                                                          \
  case X:                                                                                          \
    case_string = #C;                                                                              \
    break;
namespace {
const char* GetGraphNodeTypeString(uint32_t op) {
  const char* case_string;
  switch (static_cast<hipGraphNodeType>(op)) {
    CASE_STRING(hipGraphNodeTypeKernel, KernelNode)
    CASE_STRING(hipGraphNodeTypeMemcpy, MemcpyNode)
    CASE_STRING(hipGraphNodeTypeMemset, MemsetNode)
    CASE_STRING(hipGraphNodeTypeHost, HostNode)
    CASE_STRING(hipGraphNodeTypeGraph, GraphNode)
    CASE_STRING(hipGraphNodeTypeEmpty, EmptyNode)
    CASE_STRING(hipGraphNodeTypeWaitEvent, WaitEventNode)
    CASE_STRING(hipGraphNodeTypeEventRecord, EventRecordNode)
    CASE_STRING(hipGraphNodeTypeExtSemaphoreSignal, ExtSemaphoreSignalNode)
    CASE_STRING(hipGraphNodeTypeExtSemaphoreWait, ExtSemaphoreWaitNode)
    CASE_STRING(hipGraphNodeTypeMemAlloc, MemAllocNode)
    CASE_STRING(hipGraphNodeTypeMemFree, MemFreeNode)
    CASE_STRING(hipGraphNodeTypeMemcpyFromSymbol, MemcpyFromSymbolNode)
    CASE_STRING(hipGraphNodeTypeMemcpyToSymbol, MemcpyToSymbolNode)
    default:
      case_string = "Unknown node type";
  };
  return case_string;
};
}  // namespace

namespace hip {

hipError_t ihipGraphDebugDotPrint(hip::Graph* graph, const char* path, unsigned int flags);

std::atomic<int> GraphNode::nextID{0};
std::atomic<int> Graph::nextID{0};
std::unordered_set<GraphNode*> GraphNode::nodeSet_;
// Guards global node set
amd::Monitor GraphNode::nodeSetLock_{};
std::unordered_set<Graph*> Graph::graphSet_;
// Guards global graph set
amd::Monitor Graph::graphSetLock_{};
std::unordered_set<GraphExecBase*> GraphExecBase::graphExecSet_;
// Guards global exec graph set
// we have graphExec object as part of child graph and we need recursive lock
std::recursive_mutex GraphExecBase::graphExecSetLock_;
// Serialize the creation of internal streams from multiple threads, ensuring that each stream is
// mapped to different HSA queues.
std::recursive_mutex GraphExecBase::graphExecStreamCreateLock_;
std::shared_mutex GraphExecBase::graphExecTrimLock_;
std::unordered_set<UserObject*> UserObject::ObjectSet_;
// Guards global user object
amd::Monitor UserObject::UserObjectLock_{};
// Guards mem map add/remove against work thread
amd::Monitor GraphNode::WorkerThreadLock_{};

hipError_t GraphMemcpyNode1D::ValidateParams(void* dst, const void* src, size_t count,
                                             hipMemcpyKind kind) {
  if (dst == nullptr || src == nullptr) {
      return hipErrorInvalidValue;
  }
  if (static_cast<uint32_t>(kind) > hipMemcpyDefault && kind != hipMemcpyDeviceToDeviceNoCU) {
    return hipErrorInvalidMemcpyDirection;
  }
  size_t sOffset = 0;
  amd::Memory* srcMemory = getMemoryObjectForCurrentDevice(src, sOffset);
  size_t dOffset = 0;
  amd::Memory* dstMemory = getMemoryObjectForCurrentDevice(dst, dOffset);

  if ((srcMemory == nullptr) && (dstMemory != nullptr)) {  // host to device
    if ((kind != hipMemcpyHostToDevice) && (kind != hipMemcpyDefault)) {
      return hipErrorInvalidValue;
    }
  } else if ((srcMemory != nullptr) && (dstMemory == nullptr)) {  // device to host
    if ((kind != hipMemcpyDeviceToHost) && (kind != hipMemcpyDefault)) {
      return hipErrorInvalidValue;
    }
  }

  if (srcMemory != nullptr || dstMemory != nullptr) {
    hip::Device* dev = hip::getCurrentDevice();
    if (dev == nullptr) {
      return hipErrorInvalidDevice;
    }
    amd::Device& amdDev = *dev->devices()[0];
    if (srcMemory != nullptr) {
      hipError_t status =
          ihipMemcpy_validate_memory(amdDev, srcMemory, count, sOffset, /*read_write*/ false);
      if (status != hipSuccess) {
        return status;
      }
    }
    if (dstMemory != nullptr) {
      hipError_t status =
          ihipMemcpy_validate_memory(amdDev, dstMemory, count, dOffset, /*read_write*/ true);
      if (status != hipSuccess) {
        return status;
      }
    }
  }

  return hipSuccess;
}

// ================================================================================================
hipError_t GraphMemcpyNode::ValidateParams(const hipMemcpy3DParms* pNodeParams) {
  hipError_t status;
  status = ihipMemcpy3D_validate(pNodeParams);
  if (status != hipSuccess) {
    return status;
  }

  const HIP_MEMCPY3D pCopy = hip::getDrvMemcpy3DDesc(*pNodeParams);
  status = ihipDrvMemcpy3D_validate(&pCopy);
  if (status != hipSuccess) {
    return status;
  }
  return hipSuccess;
}

// ================================================================================================
bool Graph::isGraphValid(Graph* pGraph) {
  amd::ScopedLock lock(graphSetLock_);
  if (graphSet_.find(pGraph) == graphSet_.end()) {
    return false;
  }
  return true;
}

// ================================================================================================
void Graph::AddNode(const Node& node) {
  vertices_.emplace_back(node);
  ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_CODE, "[hipGraph] Add %s(%p)",
          GetGraphNodeTypeString(node->GetType()), node);
  node->SetParentGraph(this);
}

// ================================================================================================
void Graph::RemoveNode(const Node& node) {
  vertices_.erase(std::remove(vertices_.begin(), vertices_.end(), node), vertices_.end());
  delete node;
}

// ================================================================================================
std::vector<Node> Graph::GetRootNodes() const {
  // root nodes are all vertices with 0 in-degrees
  std::vector<Node> roots;

  for (const auto& entry : vertices_) {
    if (entry->GetInDegree() == 0) {
      roots.push_back(entry);
      ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_CODE, "[hipGraph] Root node: %s(%p)",
              GetGraphNodeTypeString(entry->GetType()), entry);
    }
  }
  return roots;
}

// ================================================================================================
// leaf nodes are all vertices with 0 out-degrees
std::vector<Node> Graph::GetLeafNodes() const {
  std::vector<Node> leafNodes;
  for (auto entry : vertices_) {
    if (entry->GetOutDegree() == 0) {
      leafNodes.push_back(entry);
    }
  }
  return leafNodes;
}

// ================================================================================================
size_t Graph::GetLeafNodeCount() const {
  int numLeafNodes = 0;
  for (auto entry : vertices_) {
    if (entry->GetOutDegree() == 0) {
      numLeafNodes++;
    }
  }
  return numLeafNodes;
}

std::vector<std::pair<Node, Node>> Graph::GetEdges() const {
  std::vector<std::pair<Node, Node>> edges;
  for (const auto& i : vertices_) {
    for (const auto& j : i->GetEdges()) {
      edges.push_back(std::make_pair(i, j));
    }
  }
  return edges;
}

// ================================================================================================
hipError_t Graph::ScheduleOneNode(Node start, int stream_id) {
  if (!start) return hipSuccess;

  // stack of pending nodes for DFS
  std::vector<Node> pending;
  pending.push_back(start);

  int sid = stream_id;

  while (!pending.empty()) {
    Node cur = pending.back();
    pending.pop_back();

    // Skip if already scheduled
    if (cur->stream_id_ != -1) {
      continue;
    }

    // Schedule current node on this branch's stream
    cur->stream_id_ = sid;

    max_streams_ = std::max(max_streams_, sid + 1);
    streams_dev_ids_[sid].insert(cur->dev_id_);

    // Process child graph separately, since there is no connection
    if (cur->GetType() == hipGraphNodeTypeGraph) {
      auto cgn   = reinterpret_cast<hip::ChildGraphNode*>(cur);
      auto child = cgn->GetChildGraph();
      hipError_t status = child->ScheduleNodes();
      if (status != hipSuccess) {
        return status;
      }
      max_streams_ = std::max(max_streams_, child->max_streams_);
    }

    const auto& edges = cur->GetEdges();
    bool end_of_branch = true;

    // To preserve left-to-right behavior, push siblings in reverse so the earlier
    // edges get processed first.
    for (int i = static_cast<int>(edges.size()) - 1; i >= 0; --i) {
      Node e = edges[static_cast<size_t>(i)];
      if (e->stream_id_ != -1) continue;
      pending.push_back(e);
      end_of_branch = false;
    }

    if (end_of_branch) {
      // Finished one depth traversal (one branch). Rotate for the next sibling/branch.
      sid = (sid + 1) % DEBUG_HIP_FORCE_GRAPH_QUEUES;
    }
  }
  return hipSuccess;
}

// ================================================================================================
hipError_t Graph::ScheduleNodes() {
  // Classic scheduling logic
  memset(&roots_[0], 0, sizeof(Node) * roots_.size());
  max_streams_ = 0;

  int stream_id = 0;
  for (auto node : vertices_) {
    if (node->stream_id_ == -1) {
      hipError_t status = ScheduleOneNode(node, stream_id);
      if (status != hipSuccess) {
        return status;
      }
      // Find the root nodes
      if ((node->GetDependencies().size() == 0) && (node->stream_id_ != 0)) {
        // Fill in only the first in the sequence
        if (roots_[node->stream_id_] == nullptr) {
          roots_[node->stream_id_] = node;
        }
      }
      // 1. Each extra root will get a new stream from the pool
      // 2. Streams will be recycled if the number of roots > streams
      stream_id = (stream_id + 1) % DEBUG_HIP_FORCE_GRAPH_QUEUES;
    }
  }

  // Topological order is needed for classic scheduling
  GraphExecBase* graphExec = dynamic_cast<GraphExecBase*>(this);
  if (graphExec && !graphExec->TopologicalOrder()) {
    ClPrint(amd::LOG_ERROR, amd::LOG_CODE, "[hipGraph] TopologicalOrder failed - invalid graph");
    return hipErrorInvalidValue;
  }

  return hipSuccess;
}

// ================================================================================================
hipError_t Graph::ScheduleNodesIntoBatches() {
  // Handle empty graph case - valid, nothing to schedule
  if (GetNodeCount() == 0) {
    return hipSuccess;
  }

  // Find execution paths hierarchically (new approach)
  GraphExecutionPaths hierarchical_paths;
  hipError_t status = FindExecutionPathsHierarchical(hierarchical_paths);
  if (status != hipSuccess) {
    return status;
  }
  if (hierarchical_paths.paths.empty()) {
    // If we have nodes but no paths, this indicates an invalid graph (likely a cycle)
    ClPrint(amd::LOG_ERROR, amd::LOG_CODE,
            "[hipGraph] No execution paths found - graph may contain cycles");
    return hipErrorInvalidValue;
  }

  // Create segments from hierarchical paths (new approach)
  status = CreateSegmentsFromPaths(hierarchical_paths);
  if (status != hipSuccess) {
    return status;
  }
  // Verify we created at least one valid segment
  if (segments_.empty()) {
    ClPrint(amd::LOG_ERROR, amd::LOG_CODE,
            "[hipGraph] No valid segments created from execution paths");
    return hipErrorInvalidValue;
  }

  // Resolve segment dependencies and calculate dependency levels
  ResolveSegmentDependencies();

  // Calculate topological order for compatibility
  // (e.g., child graphs, GetNodes() API, AutoFreeOnLaunch)
  GraphExecBase* execBase = dynamic_cast<GraphExecBase*>(this);
  if (execBase && !execBase->TopologicalOrder()) {
    ClPrint(amd::LOG_ERROR, amd::LOG_CODE,
            "[hipGraph] TopologicalOrder failed - graph may contain cycles");
    return hipErrorInvalidValue;
  }

  // A partitioner bug that places a node in two paths is invisible downstream: capture would
  // form that node's packets twice and its kernel arguments would be allocated twice, which
  // reads as a memory regression, not as a correctness failure. Equality here rules that out;
  // an inequality names it. Costs one pass over the segments at instantiate.
  {
    size_t nodes_in_segments = 0;
    std::unordered_set<Node> distinct;
    size_t kernarg = 0;
    for (const auto& seg : segments_) {
      if (seg.child_graph_ptr != nullptr) continue;
      nodes_in_segments += seg.nodes.size();
      for (Node n : seg.nodes) {
        distinct.insert(n);
        if (n != nullptr && n->GraphCaptureEnabled()) kernarg += n->GetKerArgSize();
      }
    }
    ClPrint(amd::LOG_INFO, amd::LOG_CODE,
            "[hipGraph] Partition: vertices=%zu nodes_in_segments=%zu distinct=%zu segments=%zu "
            "kernarg_bytes=%zu partition=%u",
            GetNodeCount(), nodes_in_segments, distinct.size(), segments_.size(), kernarg,
            static_cast<uint32_t>(DEBUG_HIP_GRAPH_SEGMENT_PARTITION));
    if (nodes_in_segments != distinct.size()) {
      ClPrint(amd::LOG_ERROR, amd::LOG_CODE,
              "[hipGraph] PARTITION DUPLICATES NODES: %zu placements for %zu distinct nodes -- "
              "their packets and kernel arguments are formed more than once",
              nodes_in_segments, distinct.size());
    }
  }

  ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_CODE,
          "[hipGraph] ScheduleNodesIntoBatches: Total nodes = %zu, total segments = %zu max "
          "dependency level = %d, max streams = %d",
          GetNodeCount(), segments_.size(), max_dependency_level_, max_streams_);

  return hipSuccess;
}

// ================================================================================================
// Longest-path level of every node, and the node-level interval each segment covers.
// Concurrency questions are answered here and only here: a segment's dependency_level
// counts segment hops, so two segments at different dependency levels can still be
// executing simultaneously, and comparing the two numbering schemes is a units error.
void Graph::ComputeNodeLevels() {
  node_level_.clear();
  segment_live_.assign(segments_.size(), {0, 0});

  std::vector<Node> topo;
  if (!TopologicalOrder(topo)) {
    // A cycle; the caller reports it. Leave levels flat rather than looping.
    for (Node n : vertices_) {
      if (n != nullptr) node_level_[n] = 0;
    }
    return;
  }
  for (Node n : topo) {
    int lvl = 0;
    for (Node d : n->GetDependencies()) {
      auto it = node_level_.find(d);
      if (it != node_level_.end()) lvl = std::max(lvl, it->second + 1);
    }
    node_level_[n] = lvl;
  }

  for (size_t i = 0; i < segments_.size(); ++i) {
    int lo = INT_MAX, hi = 0;
    for (Node n : segments_[i].nodes) {
      auto it = node_level_.find(n);
      if (it == node_level_.end()) continue;
      lo = std::min(lo, it->second);
      hi = std::max(hi, it->second);
    }
    if (lo == INT_MAX) lo = 0;
    segment_live_[i] = {lo, std::max(lo, hi)};
  }
}

// ================================================================================================
bool Graph::SegmentsOverlap(int a, int b) const {
  if (a < 0 || b < 0 || a >= static_cast<int>(segment_live_.size()) ||
      b >= static_cast<int>(segment_live_.size())) {
    return false;
  }
  return segment_live_[a].first <= segment_live_[b].second &&
         segment_live_[b].first <= segment_live_[a].second;
}

// ================================================================================================
int Graph::PeakLiveSegments(int dev_id) const {
  if (segment_live_.size() != segments_.size() || segments_.empty()) return 0;
  int max_level = 0;
  for (const auto& iv : segment_live_) max_level = std::max(max_level, iv.second);
  std::vector<int> delta(static_cast<size_t>(max_level) + 2, 0);
  for (size_t i = 0; i < segments_.size(); ++i) {
    if (segments_[i].dev_id != dev_id) continue;
    const int lo = segment_live_[i].first;
    const int hi = segment_live_[i].second;
    if (lo < 0 || hi < lo) continue;
    ++delta[static_cast<size_t>(lo)];
    --delta[static_cast<size_t>(hi) + 1];
  }
  int live = 0, peak = 0;
  for (int v : delta) { live += v; peak = std::max(peak, live); }
  return peak;
}

// ================================================================================================
void Graph::ResolveSegmentDependencies() {
  // Dependencies are derived from EVERY node of a segment, not just first_node.
  //
  // Reading only first_node is correct exactly while the partitioner cuts at every fork and
  // every join, because then every non-first node of a segment has a single dependency and
  // that dependency is its own predecessor inside the same segment. A mode that merges a
  // fork into one of its branches breaks that precondition, and an interior node's external
  // dependency would then vanish from the sync plan entirely -- a missing ordering edge, not
  // a lost optimisation. Deriving from all nodes removes the dependence on the precondition.
  //
  // Intra-segment edges are skipped: a segment is dispatched in order on one stream, so they
  // need no barrier and must not become self-dependencies (which would strand the segment at
  // dependency_level -1 in the topological sort below).
  size_t external_deps_from_interior = 0;

  for (size_t i = 0; i < segments_.size(); ++i) {
    auto& segment = segments_[i];

    // Use a set for O(1) duplicate detection instead of linear search on the vector
    std::unordered_set<int> dep_set(segment.segment_ids_dependencies.begin(),
                                    segment.segment_ids_dependencies.end());

    for (const auto& node : segment.nodes) {
      if (node == nullptr) continue;
      const auto& dependencies = node->GetDependencies();

      for (const auto& dep_node : dependencies) {
        // Find which segment this dependency belongs to (within this graph)
        auto dep_it = node_to_segment_id_.find(dep_node);
        if (dep_it != node_to_segment_id_.end()) {
          int dep_segment_id = dep_it->second;

          // Validate segment ID is within bounds
          if (dep_segment_id < 0 || dep_segment_id >= static_cast<int>(segments_.size())) {
            ClPrint(amd::LOG_ERROR, amd::LOG_CODE,
                    "[hipGraph] Invalid segment ID %d (segments size: %zu)",
                    dep_segment_id, segments_.size());
            continue;  // Skip invalid segment ID
          }

          // Internal to this segment: ordered by the in-order queue, no barrier needed.
          if (dep_segment_id == static_cast<int>(i)) continue;

          // Add dependency if not already present (O(1) lookup)
          if (dep_set.insert(dep_segment_id).second) {
            segment.segment_ids_dependencies.push_back(dep_segment_id);

            // Also add this segment as an edge of the dependency segment
            segments_[dep_segment_id].segment_ids_edges.push_back(i);

            if (node != segment.first_node) ++external_deps_from_interior;
          }
        }
      }
    }
  }

  // The invariant this derivation exists to stop relying on, measured rather than assumed.
  // A segment that spans neither a fork nor a join can only carry external dependencies on
  // its first node, so at partition mode 0 this prints 0. A non-zero value is exactly the
  // set of edges the segment-keyed derivation could not see.
  ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_CODE,
          "[hipGraph] ResolveSegmentDependencies: segments=%zu external_deps_from_interior=%zu",
          segments_.size(), external_deps_from_interior);

  // Recursively resolve dependencies in child graphs
  // When a parent segment depends on a segment containing a child graph node,
  // it implicitly depends on ALL segments in that child graph completing.
  for (auto& segment : segments_) {
    if (segment.child_graph_ptr != nullptr) {
      ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_CODE,
              "[hipGraph] Recursively resolving dependencies"
              "for child graph %p in segment [id=%d]",
              segment.child_graph_ptr, segment.id);

      // Child graph resolves its own internal segment dependencies
      segment.child_graph_ptr->ResolveSegmentDependencies();
    }
  }

  // Node-level intervals: the concurrency input for stream-pool sizing and assignment.
  ComputeNodeLevels();

  // Calculate dependency levels and max_streams_ using topological sort
  CalculateSegmentTopoDependencyLevels();
}

// ================================================================================================
int GraphExecSegmented::NodePos(Node n) const {
  auto it = node_pos_.find(n);
  return (it == node_pos_.end()) ? -1 : it->second;
}

// ================================================================================================
int GraphExecSegmented::HwEventSlotFor(Node n) const {
  if (n == nullptr) return -1;
  auto it = sync_plan_.node_to_hw_event.find(n);
  return (it == sync_plan_.node_to_hw_event.end()) ? -1 : it->second;
}

// ================================================================================================
// The last AQL packet belonging to |n|, or nullptr when |n| cannot carry a completion signal
// of its own. A node qualifies only if it was AQL-captured (an uncaptured node runs as an
// ordinary command and owns no packet) and its range is non-empty (an EMPTY node is a pure
// dependency point and contributes zero packets).
uint8_t* GraphExecSegmented::LastPacketOfNode(int segment_id, Node n) const {
  if (n == nullptr) return nullptr;
  if (segment_id < 0 || segment_id >= static_cast<int>(segments_.size())) return nullptr;
  auto sit = segmentBatches_.find(segment_id);
  if (sit == segmentBatches_.end()) return nullptr;

  // node_capture_status is indexed by position within the segment and is the live flag.
  // (PacketBatch::NodeRange::captured is NOT: both construction sites pass three
  // initialisers, so it is value-initialised false and read nowhere.)
  const int idx = NodePos(n);
  if (idx < 0) return nullptr;
  const auto& status = sit->second.node_capture_status;
  if (static_cast<size_t>(idx) >= status.size() || !status[idx]) return nullptr;

  for (const auto& batch : sit->second.packet_batches) {
    auto rit = batch.nodeToRangeIndex.find(n);
    if (rit == batch.nodeToRangeIndex.end()) continue;
    if (rit->second >= batch.nodeRanges.size()) return nullptr;
    const auto& r = batch.nodeRanges[rit->second];
    if (r.packetCount == 0) return nullptr;
    const size_t last = r.startIndex + r.packetCount - 1;
    if (last >= batch.dispatchPackets.size()) return nullptr;
    return batch.dispatchPackets[last];
  }
  return nullptr;
}

// ================================================================================================
// Decide, for every cross-stream segment dependency, WHICH node of the producer the consumer
// actually has to wait on, and hence which nodes need a completion slot.
//
// Waiting on the producer segment's tail is always sufficient and is what the fork/join
// partition produced; it is also what makes merging a fork into a branch pointless, because
// the fork's other branches would then wait for the whole merged branch to finish. The node
// a consumer truly depends on is the LATEST node of the producer segment any of the
// consumer's nodes has an edge from -- latest, because a segment is one in-order stream, so
// position k dominates every position below it.
void GraphExecSegmented::ComputeProducerNodes() {
  const size_t n_seg = segments_.size();
  dep_producer_.assign(n_seg, {});
  segment_producer_nodes_.assign(n_seg, {});
  node_pos_.clear();
  for (const auto& seg : segments_) {
    for (size_t i = 0; i < seg.nodes.size(); ++i) {
      node_pos_[seg.nodes[i]] = static_cast<int>(i);
    }
  }

  // A tail signal rides the segment's last packet or a barrier appended after it; an
  // interior signal has to ride the producing node's own dispatch packet. When
  // DEBUG_CLR_DEVICE_ORDERING_EDGE == 1 the completion signal is deliberately kept OFF
  // dispatch packets (so queue interceptors that rewrite dispatch packets cannot see it),
  // which interior producers cannot honour -- so they are disabled in that configuration
  // rather than silently violating it.
  const bool interior_allowed =
      (DEBUG_CLR_PP_MODE >= 1) && (DEBUG_CLR_DEVICE_ORDERING_EDGE != 1);
  const bool redirect = (DEBUG_CLR_PP_MODE >= 2) && (DEBUG_CLR_DEVICE_ORDERING_EDGE != 1);

  std::vector<std::unordered_set<Node>> seen(n_seg);
  auto add_producer = [&](int seg_id, Node n) {
    if (n != nullptr && seen[seg_id].insert(n).second) {
      segment_producer_nodes_[seg_id].push_back(n);
    }
  };

  for (size_t c = 0; c < n_seg; ++c) {
    const auto& cons = segments_[c];

    std::unordered_map<int, Node> latest;
    for (Node n : cons.nodes) {
      if (n == nullptr) continue;
      for (Node d : n->GetDependencies()) {
        auto it = node_to_segment_id_.find(d);
        if (it == node_to_segment_id_.end()) continue;
        const int p = it->second;
        if (p < 0 || p >= static_cast<int>(n_seg) || p == static_cast<int>(c)) continue;
        Node& slot = latest[p];
        if (slot == nullptr || NodePos(d) > NodePos(slot)) slot = d;
      }
    }

    for (const auto& [p, producer] : latest) {
      const auto& prod = segments_[p];
      // Same device AND same stream: the in-order queue orders them, no signal at all.
      // Producers are at a strictly lower dependency level, hence dispatched earlier.
      if (prod.dev_id == cons.dev_id && prod.stream_id == cons.stream_id) continue;

      // An interior node can only carry the signal if it owns a patchable packet.
      Node interior = nullptr;
      if (interior_allowed && producer != nullptr && producer != prod.last_node &&
          LastPacketOfNode(p, producer) != nullptr) {
        interior = producer;
      }

      const Node eff = (redirect && interior != nullptr) ? interior : prod.last_node;
      dep_producer_[c][p] = eff;
      add_producer(p, eff);
      // Mode 1 emits the interior signal but leaves consumers on the tail, so the interior
      // signal is inert. That isolates "emitting is safe" from "redirecting is correct".
      if (!redirect && interior != nullptr) add_producer(p, interior);
    }
  }

  // Leaves still need their tail signalled so EnqueueSegmentedGraph can join them back to
  // the launch stream.
  // EnqueueSegmentedGraph joins a leaf back to the launch stream through ONE signal, the
  // leaf's last_node. That is sound because a leaf segment has exactly one terminal node:
  // a segment's nodes form a chain of real graph edges, and "leaf" means no node has a
  // successor outside the segment, so the only node without a successor is the chain's end.
  // The argument survives fork merging -- but it is an argument, so it is checked. A leaf
  // with two terminal nodes would silently join on only one of them.
  size_t multi_terminal_leaves = 0;
  if (IsLeafNodeSyncRequired()) {
    for (size_t i = 0; i < n_seg; ++i) {
      if (!segments_[i].segment_ids_edges.empty()) continue;
      size_t terminals = 0;
      for (Node n : segments_[i].nodes) {
        if (n != nullptr && n->GetEdges().empty()) ++terminals;
      }
      if (terminals > 1) ++multi_terminal_leaves;
      add_producer(static_cast<int>(i), segments_[i].last_node);
    }
  }
  if (multi_terminal_leaves != 0) {
    ClPrint(amd::LOG_ERROR, amd::LOG_CODE,
            "[hipGraph] LEAF HAS MULTIPLE TERMINAL NODES: %zu leaf segment(s) -- the launch "
            "stream joins on last_node only and will not wait for the others",
            multi_terminal_leaves);
  }

  // Deterministic slot numbering: node order within the segment.
  for (size_t i = 0; i < n_seg; ++i) {
    auto& v = segment_producer_nodes_[i];
    std::stable_sort(v.begin(), v.end(),
                     [this](Node a, Node b) { return NodePos(a) < NodePos(b); });
  }
}

// ================================================================================================
Node GraphExecSegmented::ProducerFor(int consumer_seg, int producer_seg) const {
  if (consumer_seg >= 0 && consumer_seg < static_cast<int>(dep_producer_.size())) {
    auto it = dep_producer_[consumer_seg].find(producer_seg);
    if (it != dep_producer_[consumer_seg].end() && it->second != nullptr) return it->second;
  }
  if (producer_seg >= 0 && producer_seg < static_cast<int>(segments_.size())) {
    return segments_[producer_seg].last_node;
  }
  return nullptr;
}

// ================================================================================================
// Re-derive, from the materialised plan alone, the ordering each cross-segment graph edge
// actually receives, and report the ones left uncovered.
//
// This deliberately does NOT reuse dep_producer_ or the PASS 2 reduction: it reads the
// emitted waits (effective_barrier_deps_), the emitted signals (node_to_hw_event) and the
// stream assignment, and asks the question the hardware asks. Two things make an edge
// u -> v safe, and nothing else does:
//   (a) u's segment and v's segment are the same, or are on the same (device, stream) with
//       u's segment dispatched earlier -- the in-order queue orders them; or
//   (b) somewhere at or before v's segment on v's stream, a wait was emitted on a node of
//       u's segment at a position >= u's, and that node emits a completion signal.
// It can fail: dropping a wait, redirecting one to too early a node, or emitting a signal
// on a node nobody patched all show up here as a non-zero count.
size_t GraphExecSegmented::AuditSyncPlan() const {
  // Dispatch order: level by level, in segments_per_level_ order -- the same walk
  // EnqueueSegmentedGraph performs.
  std::vector<int> order;
  order.reserve(segments_.size());
  for (int level = 0; level <= max_dependency_level_; ++level) {
    auto it = segments_per_level_.find(level);
    if (it == segments_per_level_.end()) continue;
    for (int seg_id : it->second) {
      if (seg_id >= 0 && seg_id < static_cast<int>(segments_.size())) order.push_back(seg_id);
    }
  }
  std::vector<int> dispatch_rank(segments_.size(), -1);
  for (size_t i = 0; i < order.size(); ++i) dispatch_rank[order[i]] = static_cast<int>(i);

  auto stream_key = [](int dev_id, int stream_id) -> uint64_t {
    return (static_cast<uint64_t>(static_cast<uint32_t>(dev_id)) << 32) |
           static_cast<uint32_t>(stream_id);
  };

  // Replay dispatch, accumulating per stream: producer segment -> highest node position this
  // stream is known to be ordered after. Snapshot the state as each segment starts, because
  // a segment's own barriers precede its own nodes.
  std::unordered_map<uint64_t, std::unordered_map<int, int>> waited;
  std::vector<std::unordered_map<int, int>> covered_at_start(segments_.size());

  for (int seg_id : order) {
    const auto& seg = segments_[seg_id];
    auto& w = waited[stream_key(seg.dev_id, seg.stream_id)];
    for (const auto& [dep_id, pnode] : effective_barrier_deps_[seg_id]) {
      if (HwEventSlotFor(pnode) < 0) continue;  // no signal => this wait orders nothing
      const int ppos = NodePos(pnode);
      auto it = w.find(dep_id);
      if (it == w.end()) {
        w.emplace(dep_id, ppos);
      } else {
        it->second = std::max(it->second, ppos);
      }
    }
    covered_at_start[seg_id] = w;
    // Everything this segment runs is also ordered after everything its own segment's
    // in-order stream already ran, which the (a) rule below covers directly.
  }

  size_t uncovered = 0, checked = 0;
  int first_u = -1, first_v = -1;
  for (size_t v_seg = 0; v_seg < segments_.size(); ++v_seg) {
    const auto& cons = segments_[v_seg];
    for (Node v : cons.nodes) {
      if (v == nullptr) continue;
      for (Node u : v->GetDependencies()) {
        auto it = node_to_segment_id_.find(u);
        if (it == node_to_segment_id_.end()) continue;
        const int u_seg = it->second;
        if (u_seg < 0 || u_seg >= static_cast<int>(segments_.size())) continue;
        if (u_seg == static_cast<int>(v_seg)) continue;  // same segment, in-order
        ++checked;
        const auto& prod = segments_[u_seg];
        // (a) same in-order queue, producer dispatched first.
        if (prod.dev_id == cons.dev_id && prod.stream_id == cons.stream_id &&
            dispatch_rank[u_seg] >= 0 && dispatch_rank[u_seg] < dispatch_rank[v_seg]) {
          continue;
        }
        // (b) a wait on a node of u's segment at or past u.
        const auto& cov = covered_at_start[v_seg];
        auto cit = cov.find(u_seg);
        if (cit != cov.end() && cit->second >= NodePos(u)) continue;
        ++uncovered;
        if (first_u < 0) { first_u = u_seg; first_v = static_cast<int>(v_seg); }
      }
    }
  }

  if (uncovered != 0) {
    ClPrint(amd::LOG_ERROR, amd::LOG_CODE,
            "[hipGraph] SYNC AUDIT FAILED: %zu of %zu cross-segment edges have NO ordering "
            "(first: segment %d -> segment %d) -- consumers will run before their producers",
            uncovered, checked, first_u, first_v);
  }
  ClPrint(amd::LOG_INFO, amd::LOG_CODE,
          "[hipGraph] SyncAudit: cross_segment_edges=%zu uncovered=%zu", checked, uncovered);
  return uncovered;
}

// ================================================================================================
void GraphExecSegmented::BuildSyncPlan() {
  // Clean up any prior barrier packets
  for (auto* p : sync_plan_.barrier_packets) { delete[] p; }

  sync_plan_.num_segments = static_cast<int>(segments_.size());
  sync_plan_.patch_list.clear();
  sync_plan_.barrier_packets.clear();
  sync_plan_.leaf_segment_ids.clear();
  sync_plan_.node_to_hw_event.clear();
  sync_plan_.num_hw_events = 0;

  auto* device = g_devices[captureDeviceId_]->devices()[0];

  uint32_t n_completion_on_barrier = 0;
  uint32_t n_completion_on_dispatch = 0;
  uint32_t n_orphan_slots = 0;

  // PASS 0: Barrier-ROI collapse. Only runs in mode 0 (default) and only when
  // the graph is shallow (max_level<=4). Modes 1 (round-robin) and 2 (DFS)
  // never collapse. When collapse fires, every segment is folded onto stream 0
  // so the whole graph runs on the launch stream with no cross-stream barriers.
  // Init() reads collapsed_to_single_stream_ to create just one stream per device.
  collapsed_to_single_stream_ = false;
  const bool collapse_eligible = (DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING == 0) &&
                                 (max_dependency_level_ <= 4);
  if (collapse_eligible && ShouldCollapseToSingleStream()) {
    for (auto& seg : segments_) {
      seg.stream_id = 0;
      seg.needs_completion_signal = false;
    }
    collapsed_to_single_stream_ = true;
  }

  // PASS 1: Assign a compact HW-event slot only to segments whose completion
  // signal is consumed — cross-device/stream successor, or leaf when
  // leaf-sync is required. Same-stream successors are ordered by the
  // in-order queue and need no signal.
  //
  // One slot per PRODUCING NODE rather than per segment. ComputeProducerNodes applies the
  // same cross-stream/leaf criteria needs_completion_signal does, and when the partitioner
  // cuts at every fork the only producing node of a segment is its last_node, so the
  // numbering is identical to the per-segment one.
  ComputeProducerNodes();
  for (size_t i = 0; i < segments_.size(); ++i) {
    for (Node producer : segment_producer_nodes_[i]) {
      if (producer == nullptr) continue;
      if (sync_plan_.node_to_hw_event.count(producer) != 0) continue;
      sync_plan_.node_to_hw_event[producer] = sync_plan_.num_hw_events++;
    }
  }

  // PASS 2: Eliminate redundant cross-stream dependency barriers.
  //
  // Segments dispatch level-by-level and, within a level, round-robin across the
  // stream pool (see PrecomputeStreamAssignment / EnqueueSegmentedGraph), so
  // multiple segments can land on the same (device, stream). Because each stream
  // is an in-order HW queue, once an earlier-dispatched segment on a stream has
  // waited (via a barrier) for a producer's completion signal, every later
  // segment on that same stream is already ordered after that producer finishes
  // and must NOT re-wait for it. e.g. two level-1 segments that share the same
  // level-0 dependencies and land on the same stream: only the first needs the
  // dep barrier; the second inherits the ordering for free.
  //
  // ⛔ THE REDUCTION IS NODE-GRANULAR, NOT SEGMENT-GRANULAR. Memoising a bare "this stream
  // already waited on segment P" is sound only while P has exactly one externally observable
  // producer point -- its tail -- because waiting on the tail dominates all of P. Once a
  // consumer may be redirected to an INTERIOR node of P, that premise is false: an earlier
  // same-stream segment may have waited on P's node at position i while this consumer needs
  // position j > i. A bare seen-bit then drops a wait that is still required, the consumer
  // runs before its producer, and reading an unwritten pointer or gather index faults the
  // GPU with MEMORY_APERTURE_VIOLATION. So the memo stores the MAXIMUM producer-node
  // position waited on, and a dep is dropped only when that maximum already dominates the
  // position now needed. Position dominance is exactly the in-order-queue property: a
  // segment is dispatched in order on one stream, so node k implies every node below k.
  //
  // effective_barrier_deps_[seg.id] holds the (producer segment, producer node) waits each
  // segment must still emit. NOTE: a segment's id equals its position in segments_ (ids are
  // assigned sequentially at creation and pushed in order); the whole sync plan relies on it.
  effective_barrier_deps_.assign(segments_.size(), {});
  {
    // Two diagnostic settings bracket the reduction so a failure can be ATTRIBUTED to it
    // rather than argued about:
    //   3  no reduction at all -- strictly more barriers, never fewer.
    //   4  the old SEGMENT-granular reduction, deliberately restored. It is unsound with
    //      interior producers and is here only so that the fault it causes can be produced
    //      on demand and shown to disappear at mode 2. ⛔ Never a shipping setting.
    const bool dedup = (DEBUG_CLR_PP_MODE != 3);
    const bool dedup_ignores_position = (DEBUG_CLR_PP_MODE == 4);

    // Per-stream memo: producer segment -> highest producer-node position already waited on.
    // Keyed by (dev_id, stream_id) packed into one 64-bit value. Walk segments in the exact
    // dispatch order used by EnqueueSegmentedGraph.
    std::unordered_map<uint64_t, std::unordered_map<int, int>> stream_waited_deps;
    auto stream_key = [](int dev_id, int stream_id) -> uint64_t {
      return (static_cast<uint64_t>(static_cast<uint32_t>(dev_id)) << 32) |
             static_cast<uint32_t>(stream_id);
    };

    for (int level = 0; level <= max_dependency_level_; ++level) {
      auto level_it = segments_per_level_.find(level);
      if (level_it == segments_per_level_.end()) continue;

      for (int seg_id : level_it->second) {
        if (seg_id < 0 || seg_id >= static_cast<int>(segments_.size())) continue;
        const auto& seg = segments_[seg_id];
        auto& waited = stream_waited_deps[stream_key(seg.dev_id, seg.stream_id)];

        auto& reduced = effective_barrier_deps_[seg_id];
        reduced.clear();
        for (int dep_id : seg.segment_ids_dependencies) {
          if (dep_id < 0 || dep_id >= static_cast<int>(segments_.size())) continue;
          const auto& dep_seg = segments_[dep_id];
          // Same-stream/device deps are ordered by the in-order queue already.
          if (dep_seg.dev_id == seg.dev_id && dep_seg.stream_id == seg.stream_id) {
            continue;
          }
          Node pnode = ProducerFor(seg_id, dep_id);
          if (pnode == nullptr) pnode = dep_seg.last_node;
          const int ppos = NodePos(pnode);

          auto it = waited.find(dep_id);
          // ppos < 0 means the node is not in any segment, which should be impossible;
          // never dedup on it rather than treating an unknown position as dominant.
          if (dedup && it != waited.end() &&
              (dedup_ignores_position || (ppos >= 0 && it->second >= ppos))) {
            continue;
          }

          if (it == waited.end()) {
            waited.emplace(dep_id, ppos);
          } else {
            it->second = std::max(it->second, ppos);
          }
          reduced.push_back({dep_id, pnode});
        }
      }
    }
  }

  // Barrier packets are sentinel-marked with nullptr in dispatchKernelNames so that
  // activity.cpp can distinguish them from kernel/blit dispatch packets (which use "" or a
  // real name).  This avoids the empty-string ambiguity that caused the last kernel node to
  // be dropped when a copy/blit node also contributed an empty-string entry.
  static const std::string* const kBarrierKernelNamePtr = nullptr;

  // PASS 3: Materialize barrier packets and patch entries using the compact
  // hw_event slot indices computed in PASS 1.
  for (const auto& segment : segments_) {
    // Minimal cross-stream/device dependency set computed in PASS 2 (redundant
    // same-stream barriers already removed). Each entry names the producer NODE to wait on.
    const std::vector<std::pair<int, Node>>& barrier_deps = effective_barrier_deps_[segment.id];

    auto segBatchIt = segmentBatches_.find(segment.id);
    if (segBatchIt == segmentBatches_.end()) {
      continue;
    }

    auto& segBatch = segBatchIt->second;

    // Ensure at least one PacketBatch exists for barrier placement
    if (segBatch.packet_batches.empty()) {
      segBatch.packet_batches.emplace_back();
    }

    auto& firstBatch = segBatch.packet_batches[0];

    // Prepend barrier packets for segments with dependencies.
    // Optimization: when there is exactly 1 dependency and the first captured
    // packet is an ext kernel dispatch, embed the dep_signal directly into
    // that packet instead of creating a separate barrier.
    if (!barrier_deps.empty()) {
      int num_deps = static_cast<int>(barrier_deps.size());
      bool use_ext_dep = false;
      if (num_deps == 1 && !firstBatch.dispatchPackets.empty()) {
        const uint8_t* pkt = firstBatch.dispatchPackets[0];
        uint16_t first_hdr;
        memcpy(&first_hdr, pkt, sizeof(first_hdr));
        constexpr uint16_t kPktTypeMask = 0xFF;
        constexpr uint16_t kVendorSpecificType = 0;
        constexpr uint8_t kExtKernelDispatchFormat = 3;
        uint8_t amd_format = pkt[2];
        use_ext_dep = ((first_hdr & kPktTypeMask) == kVendorSpecificType)
                      && (first_hdr != 0)
                      && (amd_format == kExtKernelDispatchFormat);
      }

      if (use_ext_dep) {
        uint8_t* first_dispatch = firstBatch.dispatchPackets[0];
        // hw_event_index uses the compact slot; dep producer always has one (PASS 1).
        sync_plan_.patch_list.push_back(
            {first_dispatch, nullptr,
             HwEventSlotFor(barrier_deps[0].second),
             amd::Device::HwEventPatch::kExtDispatchDepSignal});
      } else {
        int barrier_count = (num_deps + 4) / 5;

        for (int b = 0; b < barrier_count; ++b) {
          uint8_t* barrier_pkt = device->CreateBarrierPacket();
          sync_plan_.barrier_packets.push_back(barrier_pkt);

          int start_dep = b * 5;
          int end_dep = std::min(start_dep + 5, num_deps);
          for (int d = start_dep; d < end_dep; ++d) {
            sync_plan_.patch_list.push_back(
                {barrier_pkt, nullptr,
                 HwEventSlotFor(barrier_deps[d].second),
                 d - start_dep});
          }

          firstBatch.dispatchPackets.insert(firstBatch.dispatchPackets.begin(), barrier_pkt);
          firstBatch.dispatchKernelNames.insert(firstBatch.dispatchKernelNames.begin(),
                                                kBarrierKernelNamePtr);
          firstBatch.dispatchMetadataPackets.insert(
              firstBatch.dispatchMetadataPackets.begin(), nullptr);
        }

        // nodeRanges[i].startIndex was recorded before barrier packets were prepended.
        // Update all node range indices in firstBatch to account for the inserted barriers.
        for (auto& nodeRange : firstBatch.nodeRanges) {
          nodeRange.startIndex += static_cast<size_t>(barrier_count);
        }
      }
    }

    bool last_node_uncaptured = segBatch.has_uncaptured_nodes &&
        !segment.nodes.empty() && !segBatch.node_capture_status.back();

    // Interior producers carry their own completion signal on their own last packet. The
    // patch is keyed by packet POINTER, so position within the batch is irrelevant and no
    // mid-batch insertion is needed. Without this, a consumer of an early node of a merged
    // segment would have to wait for the whole segment -- the false dependency that makes
    // merging a fork into a branch cost more than it saves.
    // LastPacketOfNode was already non-null for these nodes when PASS 1 chose them; the
    // barrier prepend above shifts startIndex and dispatchPackets by the same amount, so it
    // stays non-null. A null here would mean a slot with no emitter, i.e. a consumer that
    // waits forever, so it is counted rather than ignored.
    std::unordered_set<const uint8_t*> interior_signal_packets;
    for (Node producer : segment_producer_nodes_[segment.id]) {
      if (producer == nullptr || producer == segment.last_node) continue;
      const int islot = HwEventSlotFor(producer);
      uint8_t* pkt = LastPacketOfNode(segment.id, producer);
      if (islot < 0 || pkt == nullptr) {
        ++n_orphan_slots;
        continue;
      }
      sync_plan_.patch_list.push_back(
          {pkt, nullptr, islot, amd::Device::HwEventPatch::kCompletionSignal});
      interior_signal_packets.insert(pkt);
      ++n_completion_on_dispatch;
    }

    // hw_slot >= 0 => some consumer observes this signal (set by PASS 1).
    // Otherwise skip both the completion barrier packet and its patch entry.
    const int hw_slot = HwEventSlotFor(segment.last_node);
    const bool completion_signal_needed = (hw_slot >= 0);

    auto& lastBatch = segBatch.packet_batches.back();
    // A completion signal on a dispatch packet is exposed to queue interceptors that rewrite
    // dispatch packets; on its own barrier packet it is not.
    // ⛔ Two completion patches on ONE packet silently lose one of them: ApplyHwEventPatches
    // writes completion_signal unconditionally, so the later patch overwrites the earlier and
    // whoever waits on the overwritten signal deadlocks. The tail's carrier is the last packet
    // of the last batch, and an interior producer can OWN that packet whenever the segment's
    // own last_node contributes none -- an EMPTY node is captured but has packetCount 0. Give
    // the tail its own barrier packet in that case instead.
    const bool tail_packet_taken =
        !lastBatch.dispatchPackets.empty() &&
        interior_signal_packets.count(lastBatch.dispatchPackets.back()) != 0;
    const bool own_barrier_packet =
        last_node_uncaptured || tail_packet_taken || (DEBUG_CLR_DEVICE_ORDERING_EDGE == 1);
    if (own_barrier_packet && completion_signal_needed) {
      uint8_t* completion_barrier = device->CreateBarrierPacket();
      sync_plan_.barrier_packets.push_back(completion_barrier);

      lastBatch.dispatchPackets.push_back(completion_barrier);
      lastBatch.dispatchKernelNames.push_back(kBarrierKernelNamePtr);
      lastBatch.dispatchMetadataPackets.push_back(nullptr);

      sync_plan_.patch_list.push_back(
          {completion_barrier, nullptr, hw_slot,
           amd::Device::HwEventPatch::kCompletionSignal});
      ++n_completion_on_barrier;
    } else if (!lastBatch.dispatchPackets.empty() && completion_signal_needed) {
      // Safe to patch the last kernel dispatch directly
      uint8_t* last_pkt = lastBatch.dispatchPackets.back();
      sync_plan_.patch_list.push_back(
          {last_pkt, nullptr, hw_slot,
           amd::Device::HwEventPatch::kCompletionSignal});
      // The completion signal is embedded on this specific kernel packet. If the
      // owning node is later disabled (hipGraphNodeSetEnabled), that packet is
      // filtered out of the dispatch buffer and the signal would be lost,
      // deadlocking any consumer waiting on it. rebuildFilteredLists relocates the
      // signal to the last still-enabled packet; reserve a standalone barrier for
      // the corner case where every node packet in this batch is disabled.
      lastBatch.fallbackBarrier = device->CreateBarrierPacket();
      sync_plan_.barrier_packets.push_back(lastBatch.fallbackBarrier);
      ++n_completion_on_dispatch;
    }

    if (segment.segment_ids_edges.empty()) {
      sync_plan_.leaf_segment_ids.push_back(segment.id);
    }
  }

  // ApplyHwEventPatches indexes hw_events[patch.hw_event_index] with NO bounds check
  // (rocdevice.cpp), so a slot-space desync is a garbage hsa_signal_t written into a
  // barrier -- a hang or an aperture violation, never a diagnosable failure. Turn it into
  // one here. This can fail: a wrong producer-node mapping shows up either as a size
  // mismatch or as an out-of-range patch index.
  int bad_patches = 0;
  for (const auto& patch : sync_plan_.patch_list) {
    if (patch.hw_event_index < 0 || patch.hw_event_index >= sync_plan_.num_hw_events) {
      ++bad_patches;
    }
  }
  if (bad_patches != 0 || n_orphan_slots != 0 ||
      static_cast<int>(sync_plan_.node_to_hw_event.size()) != sync_plan_.num_hw_events) {
    ClPrint(amd::LOG_ERROR, amd::LOG_CODE,
            "[hipGraph] SyncPlan SLOT DESYNC: %zu producer nodes for %d slots, %d patch(es) "
            "outside [0,%d), %u slot(s) with no emitting packet",
            sync_plan_.node_to_hw_event.size(), sync_plan_.num_hw_events, bad_patches,
            sync_plan_.num_hw_events, n_orphan_slots);
  }

  {
    size_t prod_total = 0, prod_interior = 0, seg_multi = 0, waits = 0;
    for (size_t i = 0; i < segments_.size(); ++i) {
      const size_t k = segment_producer_nodes_[i].size();
      prod_total += k;
      if (k > 1) ++seg_multi;
      for (Node pnode : segment_producer_nodes_[i]) {
        if (pnode != nullptr && pnode != segments_[i].last_node) ++prod_interior;
      }
      waits += effective_barrier_deps_[i].size();
    }
    ClPrint(amd::LOG_INFO, amd::LOG_CODE,
            "[hipGraph] Producers: total=%zu interior=%zu segments_with_multiple=%zu "
            "emitted_waits=%zu pp_mode=%u partition=%u",
            prod_total, prod_interior, seg_multi, waits,
            static_cast<uint32_t>(DEBUG_CLR_PP_MODE),
            static_cast<uint32_t>(DEBUG_HIP_GRAPH_SEGMENT_PARTITION));
  }

  if (DEBUG_HIP_GRAPH_SYNC_AUDIT != 0) {
    AuditSyncPlan();
  }

  ClPrint(amd::LOG_INFO, amd::LOG_CODE,
          "[hipGraph] SyncPlan: segments=%d hw_events=%d completion_on_barrier=%u "
          "completion_on_dispatch=%u collapsed=%d device_ordering_edge=%u",
          sync_plan_.num_segments, sync_plan_.num_hw_events, n_completion_on_barrier,
          n_completion_on_dispatch, static_cast<int>(collapsed_to_single_stream_),
          static_cast<uint32_t>(DEBUG_CLR_DEVICE_ORDERING_EDGE));

  // Create the per-graph HW event signal pool once at instantiate time
  // (single-threaded here) and pre-create the signals, so the launch hot path
  // only pops a ready set and patches it — never creating signals.
  if (signalManager_ == nullptr) {
    signalManager_ = new GraphSignalManager();
  }
  if (sync_plan_.num_hw_events > 0) {
    // Pre-create a few sets to cover a small amount of launch overlap; the pool
    // grows on demand if more launches are concurrently in flight.
    constexpr int kPrecreatedSets = 16;
    signalManager_->Prepopulate(device, sync_plan_.num_hw_events, kPrecreatedSets);
  }

  ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_CODE,
          "[hipGraph] BuildSyncPlan: %d segments, %zu barrier packets, %d completion signals",
          sync_plan_.num_segments, sync_plan_.barrier_packets.size(), sync_plan_.num_hw_events);
}

// ================================================================================================
void Graph::CalculateSegmentTopoDependencyLevels() {
  // Topological sort of segments to calculate dependency levels
  // Assume each segment is a node and the dependencies are segments edges
  // Segments with same dependency level can be processed in parallel
  std::queue<int> queue;
  std::unordered_map<int, int> in_degree;

  // Reset max dependency level, max streams, and segments per level
  max_dependency_level_ = -1;
  max_streams_ = 1;
  segments_per_level_.clear();

  // Initialize in-degree for each segment and enqueue root segments
  for (size_t i = 0; i < segments_.size(); ++i) {
    segments_[i].dependency_level = -1;
    in_degree[i] = segments_[i].segment_ids_dependencies.size();

    if (in_degree[i] == 0) {
      // Root segments have level 0
      segments_[i].dependency_level = 0;
      queue.push(i);
      max_dependency_level_ = 0;
      segments_per_level_[0].push_back(i);
    }
  }

  // Process segments in topological order
  while (!queue.empty()) {
    int current_id = queue.front();
    queue.pop();

    auto& current_segment = segments_[current_id];
    int current_level = current_segment.dependency_level;

    // Process all segments that depend on current segment
    for (int edge_id : current_segment.segment_ids_edges) {
      auto& edge_segment = segments_[edge_id];

      // Calculate the dependency level for this segment
      // It's one level higher than the maximum of its dependencies
      int new_level = current_level + 1;
      if (edge_segment.dependency_level < new_level) {
        edge_segment.dependency_level = new_level;
        // Track the maximum dependency level
        max_dependency_level_ = std::max(max_dependency_level_, new_level);
      }

      // Decrease in-degree and enqueue if all dependencies processed
      in_degree[edge_id]--;
      if (in_degree[edge_id] == 0) {
        queue.push(edge_id);
        // Add segment to its dependency level
        segments_per_level_[edge_segment.dependency_level].push_back(edge_id);
      }
    }
  }

  // Calculate max_streams_ based on maximum parallelism at any dependency level
  for (const auto& level_segments : segments_per_level_) {
    max_streams_ = std::max(max_streams_, static_cast<int>(level_segments.second.size()));
  }

  // A segment left at level -1 never reached in-degree 0, i.e. the SEGMENT graph has a
  // cycle. It is then absent from segments_per_level_, so EnqueueSegmentedGraph never
  // dispatches it and BuildSyncPlan's PASS 2 never visits it -- a graph that silently
  // drops work and deadlocks on the missing signal. Path decomposition of a DAG cannot
  // produce this, and neither can merging a fork into a chain of its own successors, but
  // it is the failure mode any future partitioning change would hit first, so it is
  // checked rather than argued. This prints nothing when the partition is sound.
  size_t unreached = 0;
  for (const auto& seg : segments_) {
    if (seg.dependency_level < 0) ++unreached;
  }
  if (unreached != 0) {
    ClPrint(amd::LOG_ERROR, amd::LOG_CODE,
            "[hipGraph] SEGMENT CYCLE: %zu of %zu segments never reached in-degree 0 -- they "
            "will never be dispatched and every consumer of them will deadlock",
            unreached, segments_.size());
  }
}

// ================================================================================================
hipError_t Graph::FindExecutionPathsHierarchical(
    hip::Graph::GraphExecutionPaths& graph_paths) {
  graph_paths = {};
  graph_paths.graph_ptr = this;

  std::vector<Node> topological_order;
  if (!TopologicalOrder(topological_order)) {
    ClPrint(amd::LOG_ERROR, amd::LOG_CODE,
            "[hipGraph] Invalid cycle in graph or nested child graph");
    return hipErrorInvalidValue;
  }

  // Find all root nodes (nodes with no dependencies)
  const auto& root_nodes = GetRootNodes();

  std::unordered_set<unsigned int> visited;
  for (const auto& root : root_nodes) {
    // For each root, find all possible paths starting from it
    std::vector<Node> current_path;
    hipError_t status = FindPathsDFS(root, current_path, visited, graph_paths);
    if (status != hipSuccess) {
      return status;
    }
  }
  return hipSuccess;
}

// ================================================================================================
hipError_t Graph::FindPathsDFS(Node start, std::vector<Node>& current_path,
                               std::unordered_set<unsigned int>& visited,
                               hip::Graph::GraphExecutionPaths& graph_paths) {
  // Lambda to save current path as a HierarchicalPath
  auto savePath = [&graph_paths](std::vector<Node> path, int device_id,
                                  Node child_node = nullptr, int child_index = -1) {
    hip::Graph::HierarchicalPath h_path;
    h_path.nodes = std::move(path);
    h_path.device_id = device_id;
    h_path.child_graph_node = child_node;
    h_path.child_graph_paths_index = child_index;
    graph_paths.paths.push_back(std::move(h_path));
  };

  if (!start) return hipSuccess;

  // Stack of nodes to process.
  std::vector<Node> st;
  st.push_back(start);

  while (!st.empty()) {
    Node node = st.back();
    st.pop_back();
    if (!node) continue;

    // Check if already visited
    if (visited.find(node->GetID()) != visited.end()) {
      // Save any remaining path (treat visited as a branch end)
      if (!current_path.empty()) {
        int dev = current_path.back()->GetDeviceId();
        savePath(std::move(current_path), dev);
        current_path.clear();
      }
      continue;
    }

    // Mark regular nodes as visited
    visited.insert(node->GetID());

    // Check if device ID changed from previous node in path
    bool device_changed = false;
    int current_device_id = node->GetDeviceId();
    if (!current_path.empty()) {
      int prev_device_id = current_path.back()->GetDeviceId();
      if (prev_device_id != current_device_id) {
        device_changed = true;
        // Save current path before device change
        savePath(std::move(current_path), prev_device_id);
        current_path.clear();
      }
    }

    // Handle child graph nodes specially
    if (node->GetType() == hipGraphNodeTypeGraph) {
      // Save path before child graph node (if any)
      if (!current_path.empty()) {
        int dev = current_path.back()->GetDeviceId();
        savePath(std::move(current_path), dev);
        current_path.clear();
      }

      // Get the child graph and recursively process it
      auto childGraphNode = reinterpret_cast<hip::ChildGraphNode*>(node);
      auto childGraph = childGraphNode->GetChildGraph();

      if (childGraph != nullptr) {
        // Validate and discover the complete child hierarchy.
        hip::Graph::GraphExecutionPaths child_graph_exec_paths;
        hipError_t status =
            childGraph->FindExecutionPathsHierarchical(child_graph_exec_paths);
        if (status != hipSuccess) {
          return status;
        }

        // Store the child graph paths
        int child_graph_index = static_cast<int>(graph_paths.child_graph_paths.size());
        graph_paths.child_graph_paths.push_back(std::move(child_graph_exec_paths));

        // Create a path containing just the child graph node
        std::vector<Node> child_node_path = {childGraphNode};
        savePath(child_node_path, current_device_id, childGraphNode, child_graph_index);
      }

      // Clear current path and continue with edges from the child graph node
      current_path.clear();
      const auto& edges = node->GetEdges();
      for (int i = static_cast<int>(edges.size()) - 1; i >= 0; --i) {
        st.push_back(edges[static_cast<size_t>(i)]);
      }

      continue;
    }

    // Regular node - add to current path
    current_path.push_back(node);

    // Edges are out degrees, Dependencies are in degrees
    const auto& edges = node->GetEdges();
    const auto& dependencies = node->GetDependencies();

    // Check if this is a fork node (multiple outgoing edges)
    bool is_fork = edges.size() > 1;
    // Check if this is a join node (multiple incoming dependencies)
    bool is_join = dependencies.size() > 1;

    // A fork cuts its segment only because the segment's completion signal rides its LAST
    // packet, so a consumer of the fork would otherwise have to wait for everything after
    // it. Per-producer completion signals remove that reason, so at partition modes >= 1 a
    // pure fork stops cutting and is merged into one of its branches.
    //
    // A JOIN always cuts. Its dependency barriers are prepended to the segment's FIRST
    // packet batch, so a join that started mid-segment would have its wait hoisted ahead of
    // unrelated earlier work -- correct, but it would serialise the merged segment behind
    // the join's producers. Mid-batch dependency barriers would be needed to lift this;
    // they are not implemented, so the cut stands.
    const bool fork_cuts = is_fork && (DEBUG_HIP_GRAPH_SEGMENT_PARTITION == 0);

    if (fork_cuts || is_join) {
      // Save current path as a separate segment
      if (!current_path.empty()) {
        Node saved_join_node = nullptr;

        // For join nodes, save path without the join node itself
        // For fork nodes, save the complete path
        if (is_join) {
          saved_join_node = current_path.back();
          current_path.pop_back();
        }

        if (!current_path.empty()) {
          int dev = current_path.back()->GetDeviceId();
          savePath(current_path, dev);
        }
        current_path.clear();

        // For nodes that are both fork and join, save them as their own segment -- but only
        // while the fork still cuts. Otherwise the node starts a segment (it is a join) and
        // carries on into one successor rather than being isolated.
        if (saved_join_node != nullptr && fork_cuts) {
          std::vector<Node> fork_join_segment = {saved_join_node};
          savePath(std::move(fork_join_segment), saved_join_node->GetDeviceId());
        }

        // Put the join node back in current_path for further traversal
        // But not if it's also a CUTTING fork node, because we'll traverse branches separately
        if (saved_join_node != nullptr && !fork_cuts) {
          current_path.push_back(saved_join_node);
        }
      }
    }

    // Traverse the successors. Hoisted out of the cut block above because a fork that does
    // not cut still has to enqueue its successors, and pushing all edges in reverse order is
    // identical to the previous single-edge case when there is exactly one edge.
    //
    // WHICH successor continues the current path decides whether a non-cutting fork saves
    // anything. The stack pops last-pushed first, so the chosen successor is pushed LAST.
    //   mode 1: edges[0], i.e. capture order.
    //   mode 2: the successor whose fork/join-free run carries the most WORK. Work, not node
    //           count, because the boundary worth deleting is the one on the rate-determining
    //           branch; the weight is the launch thread count, the same quantity the chain
    //           assigner's node_work uses, so partitioner and assigner optimise for the same
    //           thing. MEASURED on DSV4: 269 of 271 forks have a successor that IS a join,
    //           so continuing into edges[0] blindly absorbs nothing on most of them.
    size_t continue_idx = 0;
    if (!fork_cuts && is_fork && DEBUG_HIP_GRAPH_SEGMENT_PARTITION >= 2) {
      size_t best_len = 0;
      for (size_t e = 0; e < edges.size(); ++e) {
        size_t len = 0, steps = 0;
        Node walk = edges[e];
        while (walk != nullptr && steps < 4096) {
          if (walk->GetDependencies().size() > 1) break;              // a join cuts before it
          if (visited.find(walk->GetID()) != visited.end()) break;
          size_t w = 1;
          if (walk->GetType() == hipGraphNodeTypeKernel) {
            const size_t th = static_cast<GraphKernelNode*>(walk)->GetLaunchThreadCount();
            if (th > 0) w = th;
          }
          len += w;
          ++steps;
          const auto& next = walk->GetEdges();
          if (next.size() != 1) break;                                // a fork ends the run
          walk = next[0];
        }
        if (len > best_len) { best_len = len; continue_idx = e; }
      }
    }
    for (int i = static_cast<int>(edges.size()) - 1; i >= 0; --i) {
      if (static_cast<size_t>(i) == continue_idx) continue;
      st.push_back(edges[static_cast<size_t>(i)]);
    }
    if (!edges.empty()) st.push_back(edges[continue_idx]);

    // Save any remaining path (handles leaf nodes and leaf join nodes)
    if (!current_path.empty() && edges.size() == 0) {
      int dev = current_path.back()->GetDeviceId();
      savePath(std::move(current_path), dev);
      current_path.clear();
    }
  }
  return hipSuccess;
}

// ================================================================================================
hipError_t Graph::CreateSegmentsFromPaths(
    const hip::Graph::GraphExecutionPaths& exec_paths) {
  // Clear previous segments
  segments_.clear();
  node_to_segment_id_.clear();

  // Create a segment for each execution path at this level
  int segment_id = 0;
  for (size_t i = 0; i < exec_paths.paths.size(); ++i) {
    const auto& h_path = exec_paths.paths[i];
    if (h_path.nodes.empty()) continue;

    Segment segment;
    segment.id = segment_id;
    segment.dev_id = h_path.device_id;
    segment.nodes = h_path.nodes;
    segment.first_node = h_path.nodes.front();
    segment.last_node = h_path.nodes.back();

    // Preserve child graph information from hierarchical path
    if (h_path.child_graph_node != nullptr && h_path.child_graph_paths_index >= 0) {
      // Get direct pointer to child graph from the node
      auto childGraphNode = reinterpret_cast<hip::ChildGraphNode*>(h_path.child_graph_node);
      segment.child_graph_ptr = childGraphNode->GetChildGraph();
    }

    segments_.push_back(segment);

    // Map each node in this segment to the segment ID (local to this graph)
    for (const auto& node : segment.nodes) {
      node_to_segment_id_[node] = segment_id;
      node->segment_id_ = segment_id;
    }

    segment_id++;
  }

  // Recursively process child graphs
  for (size_t i = 0; i < exec_paths.child_graph_paths.size(); ++i) {
    const auto& child_paths = exec_paths.child_graph_paths[i];

    if (child_paths.graph_ptr != nullptr) {
      // Let the child graph create its own segments
      hipError_t status = child_paths.graph_ptr->CreateSegmentsFromPaths(child_paths);
      if (status != hipSuccess) {
        return status;
      }
    }
  }
  return hipSuccess;
}

// ================================================================================================
bool Graph::TopologicalOrder(std::vector<Node>& TopoOrder) {
  std::queue<Node> q;
  std::unordered_map<Node, int> inDegree;
  for (auto entry : vertices_) {
    // Update the dependencies if a signal is required
    for (auto dep : entry->GetDependencies()) {
      // Check if the stream ID doesn't match and enable signal
      if (dep->stream_id_ != entry->stream_id_) {
        dep->signal_is_required_ = true;
      }
    }

    if (entry->GetInDegree() == 0) {
      q.push(entry);
    }
    inDegree[entry] = entry->GetInDegree();
  }
  while (!q.empty()) {
    Node node = q.front();
    TopoOrder.push_back(node);
    q.pop();
    for (auto edge : node->GetEdges()) {
      inDegree[edge]--;
      if (inDegree[edge] == 0) {
        q.push(edge);
      }
    }
  }
  if (GetNodeCount() == TopoOrder.size()) {
    return true;
  }
  return false;
}

// ================================================================================================
void Graph::clone(Graph* newGraph, bool cloneNodes) const {
  newGraph->pOriginalGraph_ = this;
  for (hip::GraphNode* entry : vertices_) {
    GraphNode* node = entry->clone();
    node->SetParentGraph(newGraph);
    newGraph->vertices_.push_back(node);
    newGraph->clonedNodes_[entry] = node;
  }

  std::vector<Node> clonedEdges;
  std::vector<Node> clonedDependencies;
  for (auto node : vertices_) {
    const std::vector<Node>& edges = node->GetEdges();
    clonedEdges.clear();
    for (auto edge : edges) {
      clonedEdges.push_back(newGraph->clonedNodes_[edge]);
    }
    newGraph->clonedNodes_[node]->SetEdges(clonedEdges);
  }
  for (auto node : vertices_) {
    const std::vector<Node>& dependencies = node->GetDependencies();
    clonedDependencies.clear();
    for (auto dep : dependencies) {
      clonedDependencies.push_back(newGraph->clonedNodes_[dep]);
    }
    newGraph->clonedNodes_[node]->SetDependencies(clonedDependencies);
  }
  for (auto& userObj : graphUserObj_) {
    userObj.first->retain();
    newGraph->graphUserObj_.insert(userObj);
    // Clone graph should have its separate graph owned ref count = 1
    newGraph->graphUserObj_[userObj.first] = 1;
    userObj.first->owning_graphs_.insert(newGraph);
  }
  // Clone the root nodes to the new graph
  // Map original root node pointers to their cloned counterparts
  if (roots_.size() > 0) {
    for (size_t i = 0; i < roots_.size(); ++i) {
      if (roots_[i] != nullptr) {
        auto it = newGraph->clonedNodes_.find(roots_[i]);
        if (it != newGraph->clonedNodes_.end()) {
          newGraph->roots_[i] = it->second;
        } else {
          newGraph->roots_[i] = nullptr;
        }
      } else {
        newGraph->roots_[i] = nullptr;
      }
    }
  }
  newGraph->memAllocNodePtrs_ = memAllocNodePtrs_;

  if (!cloneNodes) {
    newGraph->clonedNodes_.clear();
  }
}

// ================================================================================================
Graph* Graph::clone() const {
  Graph* newGraph = new Graph(getCurrentDevice());
  clone(newGraph);
  return newGraph;
}

// ================================================================================================
bool GraphExecBase::isGraphExecValid(GraphExecBase* pGraphExec) {
  std::scoped_lock lock(graphExecSetLock_);
  if (graphExecSet_.find(pGraphExec) == graphExecSet_.end()) {
    return false;
  }
  return true;
}

// ================================================================================================
hipError_t GraphExecBase::CreateStreams(uint32_t num_streams, int devId) {
  std::scoped_lock lock(graphExecStreamCreateLock_);

  if (num_streams == 0) {
    ClPrint(amd::LOG_WARNING, amd::LOG_CODE,
            "[hipGraph] Attempting to create 0 streams for device %d", devId);
    return hipSuccess;
  }

  if (devId < 0 || devId >= g_devices.size() || g_devices[devId] == nullptr) {
    ClPrint(amd::LOG_ERROR, amd::LOG_CODE, "[hipGraph] Invalid device ID %d for stream creation",
            devId);
    return hipErrorInvalidDevice;
  }

  // Check if streams already exist for this device
  if (parallel_streams_.find(devId) != parallel_streams_.end() &&
      !parallel_streams_[devId].empty()) {
    ClPrint(amd::LOG_WARNING, amd::LOG_CODE,
            "[hipGraph] Streams already exist for device %d, skipping creation", devId);
    return hipSuccess;
  }

  // num_streams is already capped by Init() but guard here defensively. On the
  // capture device the launch stream fills one slot, so create one fewer there.
  uint32_t capped = std::min(num_streams, DEBUG_HIP_FORCE_GRAPH_QUEUES);
  uint32_t max_streams = (devId == captureDeviceId_ && capped > 0) ? capped - 1 : capped;
  if (max_streams == 0) {
    return hipSuccess;
  }
  ClPrint(amd::LOG_INFO, amd::LOG_CODE, "[hipGraph] Creating %u parallel streams for device %d",
          max_streams, devId);
  parallel_streams_[devId].reserve(max_streams);
  // Track queue IDs already assigned to earlier internal streams so each new
  // stream avoids colliding with them at creation time.
  std::unordered_set<uint64_t> used_qids;
  for (uint32_t i = 0; i < max_streams; ++i) {
    auto stream = new hip::Stream(g_devices[devId], hip::Stream::Priority::Normal,
                                  hipStreamNonBlocking);

    if (!stream->Create()) {
      ClPrint(amd::LOG_ERROR, amd::LOG_CODE, "[hipGraph] Failed to create stream %u for device %d",
              i, devId);
      hip::Stream::Destroy(stream);
      for (auto& created_stream : parallel_streams_[devId]) {
        created_stream->vdev()->UnpinQueue();
        hip::Stream::Destroy(created_stream);
      }
      parallel_streams_[devId].clear();
      return hipErrorOutOfMemory;
    }

    // Pin the queue so dynamic queue management won't release it between launches
    stream->vdev()->PinQueue();
    // Acquire a queue that doesn't collide with previously created internal streams.
    // On the first stream (used_qids empty) this is a normal acquisition.
    if (!used_qids.empty()) {
      stream->vdev()->ReacquireQueueExcluding(used_qids);
    }
    used_qids.insert(stream->getQueueID());

    parallel_streams_[devId].push_back(stream);
  }
  return hipSuccess;
}

// ================================================================================================
// Creates the capture-device stream needed for cross-device launches (restores
// the full internal pool; CreateStreams() holds one slot back for same-device
// launches where the user stream fills it). Deferred to first cross-device use.
hipError_t GraphExecBase::EnsureCrossDeviceStream() {
  std::scoped_lock lock(graphExecStreamCreateLock_);

  // Callers pre-test the pointer, so a steady-state launch never takes this lock.
  if (cross_device_stream_ != nullptr) {
    return hipSuccess;
  }
  if (captureDeviceId_ < 0 || captureDeviceId_ >= g_devices.size() ||
      g_devices[captureDeviceId_] == nullptr) {
    ClPrint(amd::LOG_ERROR, amd::LOG_CODE,
            "[hipGraph] Invalid capture device ID %d for cross-device stream creation",
            captureDeviceId_);
    return hipErrorInvalidDevice;
  }

  auto stream = new hip::Stream(g_devices[captureDeviceId_], hip::Stream::Priority::Normal,
                                hipStreamNonBlocking);
  if (!stream->Create()) {
    ClPrint(amd::LOG_ERROR, amd::LOG_CODE,
            "[hipGraph] Failed to create cross-device stream for device %d", captureDeviceId_);
    hip::Stream::Destroy(stream);
    return hipErrorOutOfMemory;
  }

  auto& parallel_streams = parallel_streams_[captureDeviceId_];
  stream->vdev()->PinQueue();
  // Avoid colliding with the queues already held by the capture-device pool.
  std::unordered_set<uint64_t> used_qids;
  for (auto* existing : parallel_streams) {
    if (existing != nullptr) {
      used_qids.insert(existing->getQueueID());
    }
  }
  if (!used_qids.empty()) {
    stream->vdev()->ReacquireQueueExcluding(used_qids);
  }

  // Owned by parallel_streams_, so ~GraphExecBase() tears it down with the rest.
  parallel_streams.push_back(stream);
  cross_device_stream_ = stream;
  return hipSuccess;
}

// ================================================================================================
void GraphExecBase::FindStreamsReqPerDev() {
  // Count streams required per device based on stream-to-device mappings
  for (auto const& [stream_id, dev_ids] : streams_dev_ids_) {
    for (auto dev_id : dev_ids) {
      max_streams_dev_[dev_id]++;
    }
  }

  // Recursively process child graphs to determine their stream requirements
  for (auto node : vertices_) {
    if (node->GetType() == hipGraphNodeTypeGraph) {
      auto childNode = reinterpret_cast<ChildGraphNode*>(node);

      // Recursively find stream requirements for child graph
      childNode->FindStreamsReqPerDev();

      // Merge child graph's stream requirements with parent graph
      // Take the maximum streams needed per device to handle concurrent execution
      for (auto const& [dev_id, num_streams] : childNode->max_streams_dev_) {
        auto it = max_streams_dev_.find(dev_id);
        if (it != max_streams_dev_.end()) {
          // Device already has stream requirements - take the maximum
          max_streams_dev_[dev_id] = std::max(max_streams_dev_[dev_id], num_streams);
        } else {
          // New device - initialize with child graph's requirement
          max_streams_dev_[dev_id] = num_streams;
        }
      }
    }
  }
}

// ================================================================================================
hipError_t GraphExecSegmented::FindStreamsReqPerDevForSegments() {
  // For packet engine mode: analyze segments to determine stream requirements per device
  // We need to track the maximum number of concurrent segments per device at any level

  max_streams_dev_.clear();
  std::unordered_map<int, int> streams_per_dev_at_level;
  std::vector<GraphExecBase*> graphs_to_process{this};
  std::vector<GraphExecBase*> processed_graphs;
  std::unordered_set<GraphExecBase*> visited_graphs;

  while (!graphs_to_process.empty()) {
    GraphExecBase* graphExec = graphs_to_process.back();
    graphs_to_process.pop_back();
    if (graphExec == nullptr || !visited_graphs.insert(graphExec).second) {
      continue;
    }
    processed_graphs.push_back(graphExec);

    for (const auto& [level, segment_ids] : graphExec->segments_per_level_) {
      streams_per_dev_at_level.clear();

      // Count segments per device at this level
      for (int segment_id : segment_ids) {
        if (segment_id >= 0 && segment_id < static_cast<int>(graphExec->segments_.size())) {
          const auto& segment = graphExec->segments_[segment_id];

          streams_per_dev_at_level[segment.dev_id]++;
        }
      }

      // Update max streams per device based on this level's requirements
      for (const auto& [dev_id, count] : streams_per_dev_at_level) {
        max_streams_dev_[dev_id] = std::max(max_streams_dev_[dev_id], count);
      }
    }

    // ⛔ The per-level segment count is a LOWER BOUND on concurrency, and merging makes the
    // gap large. A merged segment absorbs its fork, so the fork's other branches become
    // dependants of the merged segment and land at the NEXT dependency level -- even though
    // they can start as soon as the fork's own node signals, which node-granular sync lets
    // them do. Sized by level population, the pool then shrinks exactly when the graph needs
    // it most, and two segments that genuinely overlap get the same stream and serialise.
    // Size from node-level LIVENESS instead, which counts segments that can be executing at
    // the same instant. Gated: at partition mode 0 this would also change stock sizing,
    // which is a separate question and must not ride along on the default path.
    if (DEBUG_HIP_GRAPH_SEGMENT_PARTITION != 0) {
      for (const auto& [dev_id, count] : streams_per_dev_at_level) {
        (void)count;
        const int live = graphExec->PeakLiveSegments(dev_id);
        max_streams_dev_[dev_id] = std::max(max_streams_dev_[dev_id], live);
      }
    }

    {
      // Starts-vs-live at a glance. Equal means the old lower bound was tight here and this
      // change buys nothing; live > starts is exactly the concurrency merging was losing.
      int max_start = 0;
      for (const auto& [lvl, ids] : graphExec->segments_per_level_) {
        (void)lvl;
        max_start = std::max(max_start, static_cast<int>(ids.size()));
      }
      std::set<int> devs;
      for (const auto& s : graphExec->segments_) devs.insert(s.dev_id);
      for (int d : devs) {
        ClPrint(amd::LOG_INFO, amd::LOG_CODE,
                "[hipGraph] Concurrency: dev=%d peak_starts_any_level=%d peak_live=%d "
                "segments=%zu partition=%u",
                d, max_start, graphExec->PeakLiveSegments(d), graphExec->segments_.size(),
                static_cast<uint32_t>(DEBUG_HIP_GRAPH_SEGMENT_PARTITION));
      }
    }

    for (const auto& segment : graphExec->segments_) {
      if (segment.child_graph_ptr != nullptr) {
        auto childGraphExec = dynamic_cast<GraphExecBase*>(segment.child_graph_ptr);
        if (childGraphExec != nullptr) {
          graphs_to_process.push_back(childGraphExec);
        }
      }
    }
  }

  // Validate only after all nested child graphs have contributed their devices.
  if (max_streams_dev_.size() > 1) {
    ClPrint(amd::LOG_ERROR, amd::LOG_CODE,
            "[hipGraph] Multi-device graph is not supported on the segment scheduling path");
    captureDeviceId_ = -1;
    return hipErrorNotSupported;
  }

  const int capture_device_id =
      max_streams_dev_.empty() ? hip::getCurrentDevice()->deviceId()
                               : max_streams_dev_.begin()->first;
  for (GraphExecBase* graph_exec : processed_graphs) {
    graph_exec->captureDeviceId_ = capture_device_id;
  }
  return hipSuccess;
}

// ================================================================================================
// max_streams_dev_ represents the total stream-slot count uniformly per device;
// CreateStreams() does not adjust it. The capture device's slot 0 instead comes
// from the launch stream (same-device) or EnsureCrossDeviceStream() (cross-device).
// Init() caps this against DEBUG_HIP_FORCE_GRAPH_QUEUES before assignment runs.
size_t GraphExecSegmented::GetStreamPoolSize(int dev_id) const {
  auto it = max_streams_dev_.find(dev_id);
  return (it != max_streams_dev_.end() && it->second > 0) ? static_cast<size_t>(it->second) : 1;
}

// ================================================================================================
// Shared placement core for every level-framed strategy.
//
// Two segments conflict when giving them the same stream would serialise work that could
// otherwise overlap. Which segments those are depends on the partition:
//   partition 0  same dependency level. Every segment is one fork/join-free run, so a
//                segment's whole lifetime is its level, and this is what stock round-robin
//                assumes. Reproduced exactly, so the default path is untouched.
//   partition >0 overlapping node-level intervals. A merged segment absorbs its fork, so
//                the fork's other branches sit at the NEXT dependency level while running
//                CONCURRENTLY with it; level equality no longer describes concurrency and
//                using it hands them the same stream.
void GraphExecSegmented::ComputeRoundRobinAssignment(std::vector<int>& out) const {
  out.assign(segments_.size(), -1);
  const bool by_interval = (DEBUG_HIP_GRAPH_SEGMENT_PARTITION != 0);

  for (int level = 0; level <= max_dependency_level_; ++level) {
    auto it = segments_per_level_.find(level);
    if (it == segments_per_level_.end()) continue;

    for (int seg_id : it->second) {
      if (seg_id < 0 || seg_id >= static_cast<int>(segments_.size())) continue;
      const auto& seg = segments_[seg_id];
      const size_t pool = GetStreamPoolSize(seg.dev_id);
      std::vector<size_t> load(pool, 0);
      for (size_t k = 0; k < segments_.size(); ++k) {
        if (static_cast<int>(k) == seg_id || out[k] < 0) continue;
        if (segments_[k].dev_id != seg.dev_id) continue;
        const bool conflict = by_interval
            ? SegmentsOverlap(static_cast<int>(k), seg_id)
            : (segments_[k].dependency_level == seg.dependency_level);
        if (!conflict) continue;
        if (static_cast<size_t>(out[k]) < pool) ++load[static_cast<size_t>(out[k])];
      }
      // Lowest-index minimum. At partition 0 the conflict set is exactly the segments
      // already placed at this level, so this reproduces the rotating per-level counter
      // stock round-robin used, slot for slot.
      size_t best = 0;
      for (size_t s = 1; s < pool; ++s) {
        if (load[s] < load[best]) best = s;
      }
      out[seg_id] = static_cast<int>(best);
    }
  }
}

// ================================================================================================
void GraphExecSegmented::RoundRobinStreamAssignment() {
  std::vector<int> assignment;
  ComputeRoundRobinAssignment(assignment);
  // Only entries a level actually covered are applied, preserving the previous behaviour of
  // leaving an unlevelled segment's stream_id untouched.
  for (size_t i = 0; i < segments_.size(); ++i) {
    if (assignment[i] >= 0) segments_[i].stream_id = assignment[i];
  }

  ComputeCompletionSignalFlags();
}

// ================================================================================================
// Chain-following assignment (DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING=4).
//
// The placement frame above is kept and only the choice within it changes: the slot is always
// one of minimal conflict load, so the multiset of per-slot loads is exactly round-robin's and
// only the labels are permuted -- concurrency structure, and with it stream and ring skew, is
// unchanged. Within that freedom each segment prefers the slot of the dependency it waits
// longest for, the edge that would otherwise fall on the critical path. Segments choose in
// descending critical-path order, so a preference is refused only in favour of a longer chain.
void GraphExecSegmented::ChainAffinityStreamAssignment() {
  std::vector<size_t> seg_work;
  std::vector<size_t> cp;
  ComputeSegmentWorkAndCriticalPath(seg_work, cp);

  for (auto& seg : segments_) {
    seg.stream_id = -1;
  }
  const bool by_interval = (DEBUG_HIP_GRAPH_SEGMENT_PARTITION != 0);

  for (int level = 0; level <= max_dependency_level_; ++level) {
    auto it = segments_per_level_.find(level);
    if (it == segments_per_level_.end()) continue;

    std::vector<int> order;
    order.reserve(it->second.size());
    for (int seg_id : it->second) {
      if (seg_id >= 0 && seg_id < static_cast<int>(segments_.size())) order.push_back(seg_id);
    }
    std::stable_sort(order.begin(), order.end(),
                     [&cp](int a, int b) { return cp[a] > cp[b]; });

    for (int seg_id : order) {
      auto& seg = segments_[seg_id];
      const size_t pool = GetStreamPoolSize(seg.dev_id);
      std::vector<size_t> load(pool, 0);
      for (size_t k = 0; k < segments_.size(); ++k) {
        if (static_cast<int>(k) == seg_id || segments_[k].stream_id < 0) continue;
        if (segments_[k].dev_id != seg.dev_id) continue;
        const bool conflict = by_interval
            ? SegmentsOverlap(static_cast<int>(k), seg_id)
            : (segments_[k].dependency_level == seg.dependency_level);
        if (!conflict) continue;
        if (static_cast<size_t>(segments_[k].stream_id) < pool) {
          ++load[static_cast<size_t>(segments_[k].stream_id)];
        }
      }

      // The producer this segment waits longest for. Dependencies sit at strictly lower
      // dependency levels and are therefore already assigned; anything unassigned, or on
      // another device, is skipped rather than guessed.
      int want = -1;
      size_t want_cp = 0;
      for (int dep_id : seg.segment_ids_dependencies) {
        if (dep_id < 0 || dep_id >= static_cast<int>(segments_.size())) continue;
        const auto& dep = segments_[dep_id];
        if (dep.dev_id != seg.dev_id || dep.stream_id < 0) continue;
        if (want < 0 || cp[dep_id] > want_cp) {
          want = dep.stream_id;
          want_cp = cp[dep_id];
        }
      }

      size_t min_load = load[0];
      for (size_t s = 1; s < pool; ++s) min_load = std::min(min_load, load[s]);

      size_t chosen = 0;
      if (want >= 0 && static_cast<size_t>(want) < pool && load[want] == min_load) {
        chosen = static_cast<size_t>(want);
      } else {
        for (size_t s = 0; s < pool; ++s) {
          if (load[s] == min_load) { chosen = s; break; }
        }
      }
      seg.stream_id = static_cast<int>(chosen);
    }
  }

  ComputeCompletionSignalFlags();
}

// ================================================================================================
// One line per instantiate naming the strategy that actually ran and what it did.
// Sole printer: an arm is verified from this line, never from the env var having been set.
//   moved_vs_rr  segments whose stream differs from round-robin's. 0 by construction for the
//                round-robin modes; a 0 on a heuristic mode means it never disagreed and the
//                arm cannot show an effect.
//   invalid      segments left outside [0, pool). Non-zero is a bug, not a configuration:
//                resolveSegmentStream's `stream_id % streams.size()` turns both -1 and an
//                over-range id into a valid-looking index, so nothing downstream can fail.
void GraphExecSegmented::LogStreamAssignment(uint32_t requested, const char* effective) const {
  std::vector<int> rr;
  ComputeRoundRobinAssignment(rr);

  int moved_vs_rr = 0;
  int invalid = 0;
  std::map<int, std::vector<int>> per_stream;  // dev_id -> segment count per slot

  for (size_t i = 0; i < segments_.size(); ++i) {
    const auto& seg = segments_[i];
    const size_t pool = GetStreamPoolSize(seg.dev_id);
    auto& counts = per_stream[seg.dev_id];
    if (counts.size() < pool) counts.resize(pool, 0);

    if (seg.stream_id < 0 || static_cast<size_t>(seg.stream_id) >= pool) {
      ++invalid;
    } else {
      ++counts[seg.stream_id];
    }
    if (rr[i] >= 0 && rr[i] != seg.stream_id) ++moved_vs_rr;
  }

  // The property the shared placement frame is supposed to buy: the multiset of per-slot
  // loads within each conflict set equals round-robin's -- same concurrency structure, labels
  // permuted. Reported per instantiate because it is the safety argument, and it can fail.
  int perm_mismatch_levels = 0;
  for (int level = 0; level <= max_dependency_level_; ++level) {
    auto lit = segments_per_level_.find(level);
    if (lit == segments_per_level_.end()) continue;
    std::map<int, std::vector<int>> got, want;
    for (int seg_id : lit->second) {
      if (seg_id < 0 || seg_id >= static_cast<int>(segments_.size())) continue;
      const auto& seg = segments_[seg_id];
      const size_t pool = GetStreamPoolSize(seg.dev_id);
      auto& g = got[seg.dev_id];
      auto& w = want[seg.dev_id];
      if (g.size() < pool) g.resize(pool, 0);
      if (w.size() < pool) w.resize(pool, 0);
      if (seg.stream_id >= 0 && static_cast<size_t>(seg.stream_id) < pool) ++g[seg.stream_id];
      if (rr[seg_id] >= 0 && static_cast<size_t>(rr[seg_id]) < pool) ++w[rr[seg_id]];
    }
    for (auto& [dev_id, g] : got) {
      auto& w = want[dev_id];
      std::sort(g.begin(), g.end());
      std::sort(w.begin(), w.end());
      if (g != w) { ++perm_mismatch_levels; break; }
    }
  }

  std::string detail;
  for (const auto& [dev_id, counts] : per_stream) {
    detail += " dev" + std::to_string(dev_id) + ":pool=" + std::to_string(counts.size()) +
              " per_stream=[";
    for (size_t s = 0; s < counts.size(); ++s) {
      if (s) detail += ",";
      detail += std::to_string(counts[s]);
    }
    detail += "]";
  }

  ClPrint(amd::LOG_INFO, amd::LOG_CODE,
          "[hipGraph] StreamAssignment: requested=%u effective=%s partition=%u segs=%zu "
          "max_level=%d%s moved_vs_rr=%d invalid=%d perm_mismatch_levels=%d",
          requested, effective, static_cast<uint32_t>(DEBUG_HIP_GRAPH_SEGMENT_PARTITION),
          segments_.size(), max_dependency_level_, detail.c_str(), moved_vs_rr, invalid,
          perm_mismatch_levels);

  if (invalid != 0) {
    ClPrint(amd::LOG_ERROR, amd::LOG_CODE,
            "[hipGraph] StreamAssignment: %d segment(s) outside [0,pool) after strategy '%s' -- "
            "these alias onto an arbitrary stream instead of failing",
            invalid, effective);
  }
}

// ================================================================================================
// ⛔ "Does this segment need a completion signal" is the wrong question once a segment may
// contain several nodes with cross-stream consumers: the answer is a COUNT, not a flag, and
// it is a property of NODES. The count matters because ShouldCollapseToSingleStream feeds it
// into signal_est, which is a DECISION input, not a log line -- a per-segment answer
// undercounts a merged graph's sync cost and biases the collapse verdict.
// needs_completion_signal is kept as `count > 0` for the existing readers.
void GraphExecSegmented::ComputeCompletionSignalFlags() {
  const bool leaf_sync_required = IsLeafNodeSyncRequired();
  for (auto& seg : segments_) {
    seg.needs_completion_signal = false;
    seg.producer_node_count = 0;

    if (seg.segment_ids_edges.empty()) {
      // Leaf segments need a completion signal so EnqueueSegmentedGraph can
      // sync them back to the launch stream via graph_accumulate dep_signals.
      if (leaf_sync_required) {
        seg.needs_completion_signal = true;
        seg.producer_node_count = 1;
      }
      continue;
    }

    // Distinct nodes of this segment that some node on another (device, stream) depends on.
    std::unordered_set<Node> producers;
    for (Node n : seg.nodes) {
      if (n == nullptr) continue;
      for (Node succ : n->GetEdges()) {
        auto it = node_to_segment_id_.find(succ);
        if (it == node_to_segment_id_.end()) continue;
        const int c = it->second;
        if (c < 0 || c >= static_cast<int>(segments_.size()) || c == seg.id) continue;
        const auto& cons = segments_[c];
        if (cons.dev_id != seg.dev_id || cons.stream_id != seg.stream_id) {
          producers.insert(n);
          break;
        }
      }
    }
    seg.producer_node_count = static_cast<int>(producers.size());
    seg.needs_completion_signal = !producers.empty();
  }
}

// ================================================================================================
// DFS-based stream assignment for segment DAG.
// Modeled after the classic path's ScheduleOneNode traversal pattern:
//   - linear chains stay on the same stream
//   - sid rotates at leaf segments (end of branch), not at forks
//   - DFS is started from every unscheduled segment (not just dependency-free
//     roots), with sid incrementing once per outer-loop iteration
// The pool is max_streams_dev_, the same source every other strategy uses; Init() has
// already capped it against DEBUG_HIP_FORCE_GRAPH_QUEUES.
void GraphExecSegmented::DFSStreamAssignment() {
  // Reset all stream IDs
  for (auto& seg : segments_) {
    seg.stream_id = -1;
  }

  int sid = 0;

  // Mirrors ScheduleNodes(): iterate all segments, start a new DFS for each
  // unscheduled one with the current sid, then increment sid for the next entry.
  for (int i = 0; i < static_cast<int>(segments_.size()); ++i) {
    if (segments_[i].stream_id != -1) continue;

    // Determine pool size for this entry's device
    int pool = static_cast<int>(GetStreamPoolSize(segments_[i].dev_id));

    // Stack carries segment ids — sid is shared across the entire DFS from this
    // entry point, exactly like ScheduleOneNode's single `sid` variable.
    std::vector<int> pending;
    pending.push_back(i);

    while (!pending.empty()) {
      int cur_id = pending.back();
      pending.pop_back();

      if (cur_id < 0 || cur_id >= static_cast<int>(segments_.size())) continue;
      auto& cur = segments_[cur_id];

      if (cur.stream_id != -1) continue;

      cur.stream_id = sid % pool;

      // Push unassigned successors in reverse order (preserve left-to-right)
      bool end_of_branch = true;
      for (int j = static_cast<int>(cur.segment_ids_edges.size()) - 1; j >= 0; --j) {
        int edge_id = cur.segment_ids_edges[j];
        if (edge_id >= 0 && edge_id < static_cast<int>(segments_.size()) &&
            segments_[edge_id].stream_id == -1) {
          pending.push_back(edge_id);
          end_of_branch = false;
        }
      }

      // Rotate sid at leaf — mirrors ScheduleOneNode exactly
      if (end_of_branch) {
        sid = (sid + 1) % pool;
      }
    }

    // Mirrors ScheduleNodes()'s stream_id = (stream_id+1) % pool after each entry
    sid = (sid + 1) % pool;
  }

  ComputeCompletionSignalFlags();
}

// ================================================================================================
// Select stream assignment algorithm (DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING):
//   0 = Collapse (with existing ROI check + max_level<=4 guard) then round-robin
//   1 = Round-robin only, no collapse
//   2 = DFS only, no collapse
//   4 = Chain affinity: round-robin's placement frame, critical-predecessor preference
void GraphExecSegmented::SelectStreamAssignment() {
  const uint32_t requested = DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING;
  const char* effective = nullptr;

  if (requested == 1) {
    effective = "roundrobin";
    RoundRobinStreamAssignment();
  } else if (requested == 2) {
    effective = "dfs";
    DFSStreamAssignment();
  } else if (requested == 4) {
    effective = "chain";
    ChainAffinityStreamAssignment();
  } else {
    // 0 = collapse-eligible (ROI heuristic + max_level<=4) then round-robin.
    // ShouldCollapseToSingleStream() is called later in BuildSyncPlan; the extra
    // max_level guard there prevents collapse on deep graphs where multi-stream
    // overlap is genuinely valuable.
    if (requested != 0) {
      static std::once_flag once;
      std::call_once(once, [requested]() {
        ClPrint(amd::LOG_ERROR, amd::LOG_CODE,
                "[hipGraph] DEBUG_HIP_GRAPH_SEGMENT_SCHEDULING=%u is not a mode; behaving as 0",
                requested);
      });
    }
    effective = (max_dependency_level_ > 4) ? "hybrid-rr" : "hybrid-collapse-eligible";
    RoundRobinStreamAssignment();
  }

  LogStreamAssignment(requested, effective);
}

// ================================================================================================
// Barrier-ROI heuristic.
//
// Multi-stream segment scheduling only pays off when the device-side overlap it
// unlocks exceeds the cost of the cross-stream barriers/signals it requires. For
// launch-overhead-bound graphs (many tiny kernels, near-serial dependencies) the
// barriers have a high overhead: collapsing every segment onto one in-order stream
// removes them entirely and is measurably faster on both the launch and the
// instantiate path.
//
// Both sides are estimated in cheap structural units, with no device timing:
//   * barrier_est   = number of barrier *packets* multi-stream would emit. This
//                     mirrors BuildSyncPlan's materialization rather than the raw
//                     dependency count: a single barrier packet resolves up to 5
//                     dependencies ((deps + 4) / 5 packets), and a lone
//                     dependency is folded into the segment's ext kernel-dispatch
//                     packet with no separate barrier at all. SelectStream-
//                     Assignment() has already run by the time this is called, so
//                     only *cross-stream/device* deps are counted (same-stream
//                     deps are ordered by the in-order queue and emit no barrier),
//                     matching what PASS 2/PASS 3 of BuildSyncPlan actually do.
//   * signal_est    = number of completion *signals* multi-stream would emit.
//                     Reuses each segment's needs_completion_signal flag (set by
//                     SelectStreamAssignment with the same cross-stream/leaf
//                     criteria BuildSyncPlan uses), so producers whose consumers
//                     share their stream are not over-counted. Signals are real
//                     per-launch host cost (signal-pool acquire + packet patch)
//                     that the barrier count alone misses, so they belong in the
//                     ROI denominator alongside barriers.
//   * parallel_slack = total work - critical-path work. The work that is
//                      genuinely off the longest dependency chain and could
//                      therefore overlap on another stream. Work is measured in
//                      machine-occupancy passes, not node count: each kernel
//                      weighs ceil(launch_threads / machine_threads) (>=1), so a
//                      kernel that fills the GPU once is 1 unit and one needing N
//                      passes is N. Non-kernel nodes and sub-machine launches are
//                      1 unit, preserving the original node-count behaviour for
//                      the small launch-bound kernels the gate targets. This is
//                      what keeps two *long-running* independent kernels multi-
//                      stream (their slack outgrows the threshold) while tiny
//                      independent kernels (e.g. PyFR's stubs) still collapse.
//
// Collapse when parallel_slack < min_overlap * (barrier_est + signal_est): i.e.
// keep multi-stream only when each unit of cross-stream sync overhead buys at
// least min_overlap nodes of overlappable work. Folding collapse onto the launch
// stream removes both barriers and signals (it runs inline, 0/0), so the full
// sync cost is what multi-stream genuinely pays over collapse. Tunable via
// DEBUG_HIP_GRAPH_MIN_OVERLAP; 0 disables the gate.
// ================================================================================================
// Structural work per segment, and the critical-path work ending at each segment.
//
// Both are pure functions of the segment DAG and node launch geometry: no device timing and --
// load bearing -- no read of stream_id, so this is valid before stream assignment as well as
// after. Work is in machine-occupancy passes: a kernel that fills the GPU once weighs 1, one
// needing N passes weighs N, and every non-kernel or sub-machine launch weighs 1.
// Unconditional: callers that gate on their own preconditions apply them themselves.
void GraphExecSegmented::ComputeSegmentWorkAndCriticalPath(std::vector<size_t>& work,
                                                           std::vector<size_t>& cp) const {
  work.assign(segments_.size(), 0);
  cp.assign(segments_.size(), 0);
  if (segments_.empty()) return;

  size_t machine_threads = 0;
  const int dev0 = segments_.front().dev_id;
  if (dev0 >= 0) {
    const auto& dinfo = g_devices[dev0]->devices()[0]->info();
    machine_threads = static_cast<size_t>(dinfo.maxComputeUnits_) * dinfo.maxThreadsPerCU_;
  }
  auto node_work = [machine_threads](Node n) -> size_t {
    if (n == nullptr || n->GetType() != hipGraphNodeTypeKernel) return 1;
    const size_t threads = static_cast<GraphKernelNode*>(n)->GetLaunchThreadCount();
    if (threads == 0 || machine_threads == 0) return 1;
    return std::max<size_t>(1, (threads + machine_threads - 1) / machine_threads);
  };

  for (size_t i = 0; i < segments_.size(); ++i) {
    for (Node n : segments_[i].nodes) work[i] += node_work(n);
  }

  // Level order guarantees every dependency is final before it is read.
  for (int level = 0; level <= max_dependency_level_; ++level) {
    auto it = segments_per_level_.find(level);
    if (it == segments_per_level_.end()) continue;
    for (int seg_id : it->second) {
      if (seg_id < 0 || seg_id >= static_cast<int>(segments_.size())) continue;
      size_t best_dep = 0;
      for (int dep_id : segments_[seg_id].segment_ids_dependencies) {
        if (dep_id < 0 || dep_id >= static_cast<int>(segments_.size())) continue;
        best_dep = std::max(best_dep, cp[dep_id]);
      }
      cp[seg_id] = best_dep + work[seg_id];
    }
  }
}

// ================================================================================================
bool GraphExecSegmented::ShouldCollapseToSingleStream() const {
  const uint32_t min_overlap = DEBUG_HIP_GRAPH_MIN_OVERLAP;
  if (min_overlap == 0) return false;            // gate disabled
  if (segments_.size() < 2) return false;        // nothing to parallelize

  const int dev0 = segments_.front().dev_id;
  for (const auto& seg : segments_) {
    if (seg.dev_id != dev0) return false;
  }

  // The sync-cost half DOES read stream_id, so unlike the work terms it is only meaningful
  // once SelectStreamAssignment has run -- which it has: this is called from BuildSyncPlan.
  size_t barrier_est = 0;
  size_t signal_est = 0;
  for (size_t i = 0; i < segments_.size(); ++i) {
    const auto& seg = segments_[i];
    size_t cross_deps = 0;
    for (int dep_id : seg.segment_ids_dependencies) {
      if (dep_id < 0 || dep_id >= static_cast<int>(segments_.size())) continue;
      const auto& dep_seg = segments_[dep_id];
      if (dep_seg.dev_id != seg.dev_id || dep_seg.stream_id != seg.stream_id) {
        ++cross_deps;
      }
    }
    if (cross_deps >= 2) {
      barrier_est += (cross_deps + 4) / 5;
    }
    // One signal per PRODUCING NODE, not per segment: a merged segment can carry several.
    signal_est += static_cast<size_t>(std::max(0, seg.producer_node_count));
  }
  const size_t sync_cost = barrier_est + signal_est;
  if (sync_cost == 0) return false;

  // total_work is a sum over ALL segments and still is. critical_path_work was a max over
  // level-visited segments only; it is now a max over all, and unvisited entries are 0, so
  // the max is unchanged.
  std::vector<size_t> seg_work;
  std::vector<size_t> cp;
  ComputeSegmentWorkAndCriticalPath(seg_work, cp);
  size_t total_work = 0;
  size_t critical_path_work = 0;
  for (size_t i = 0; i < segments_.size(); ++i) {
    total_work += seg_work[i];
    critical_path_work = std::max(critical_path_work, cp[i]);
  }

  const size_t parallel_slack =
      (total_work > critical_path_work) ? (total_work - critical_path_work) : 0;
  const bool collapse = parallel_slack < static_cast<size_t>(min_overlap) * sync_cost;

  ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_CODE,
          "[hipGraph] Single-stream gate: work=%zu critical_path=%zu slack=%zu "
          "barrier_packets~=%zu signals~=%zu sync_cost=%zu min_overlap=%u -> %s",
          total_work, critical_path_work, parallel_slack, barrier_est, signal_est,
          sync_cost, min_overlap,
          collapse ? "collapse to single stream" : "keep multi-stream");
  return collapse;
}

// Carries the per-launch state needed by the completion callback: the graph
// whose refcount to drop, plus the signal set (and its device) to re-arm and
// return to the pool now that the launch's GPU work is done.
struct GraphLaunchCleanup {
  GraphExecBase* exec;
  amd::Device* device;
  std::vector<void*> signal_set;
};

// ================================================================================================
// GraphExecClassic — PAL/Windows path: classic topological enqueue, no AQL capture.
// ================================================================================================
hipError_t GraphExecClassic::Init() {
  hipError_t status = hipSuccess;

  // ScheduleNodes does DFS stream assignment + TopologicalOrder
  status = ScheduleNodes();
  if (status != hipSuccess) {
    return status;
  }

  if (max_streams_ >= 1) {
    FindStreamsReqPerDev();

    if (max_streams_dev_.size() == 1) {
      captureDeviceId_ = max_streams_dev_.begin()->first;
    } else if (max_streams_dev_.size() > 1) {
      ClPrint(amd::LOG_ERROR, amd::LOG_CODE,
              "[hipGraph] Multi-device graph is not supported for classic scheduling path");
      captureDeviceId_ = -1;
      return hipErrorNotSupported;
    }
    for (auto& [dev_id, count] : max_streams_dev_) {
      count = std::min(count, static_cast<int>(DEBUG_HIP_FORCE_GRAPH_QUEUES));
    }

    for (auto const& [dev_id, num_streams] : max_streams_dev_) {
      if (num_streams > 0) {
        status = CreateStreams(num_streams, dev_id);
        if (status != hipSuccess) {
          return status;
        }
      }
    }
  }

  return status;
}

// ================================================================================================
hipError_t GraphExecClassic::Run(hip::Stream* launch_stream) {
  hipError_t status = hipSuccess;

  {
    std::shared_lock<std::shared_mutex> trim_guard(graphExecTrimLock_);
    this->retain();
  }

  Node firstNode = topoOrder_.empty() ? nullptr : topoOrder_[0];

  if (flags_ & hipGraphInstantiateFlagAutoFreeOnLaunch) {
    if (firstNode != nullptr) {
      auto* parentGraph = firstNode->GetParentGraph();
      auto* pool = parentGraph->Device()->GetGraphMemoryPool();
      for (auto* node : topoOrder_) {
        if (node->GetType() == hipGraphNodeTypeMemAlloc) {
          static_cast<GraphMemAllocNode*>(node)->ReleaseCachedMapping(pool, launch_stream);
        }
      }
      parentGraph->FreeAllMemory(launch_stream);
      parentGraph->memalloc_nodes_ = 0;
      if (!AMD_DIRECT_DISPATCH) {
        launch_stream->finish();
      }
    }
  }

  if (repeatLaunch_ == true) {
    if (firstNode != nullptr && firstNode->GetParentGraph()->GetMemAllocNodeCount() > 0) {
      this->release();
      return hipErrorInvalidValue;
    }
  } else {
    repeatLaunch_ = true;
  }

  ClPrint(amd::LOG_DEBUG, amd::LOG_CODE, "GraphExecClassic::Run max_streams: %d, on device: %d",
          max_streams_, launch_stream->DeviceId());

  launch_stream->vdev()->SetPreferredQueue();
  launch_stream->vdev()->AcquireQueueWithPreference();
  UpdateStreams(launch_stream);

  if (max_streams_ == 1 && captureDeviceId_ != launch_stream->DeviceId()) {
    for (int i = 0; i < topoOrder_.size(); i++) {
      topoOrder_[i]->SetStream(launch_stream);
      status = topoOrder_[i]->CreateCommand(topoOrder_[i]->GetQueue());
      if (status != hipSuccess) {
        this->release();
        return status;
      }
      status = topoOrder_[i]->EnqueueCommands(launch_stream);
      if (status != hipSuccess) {
        this->release();
        return status;
      }
    }
  } else {
    // Execute all nodes in the graph
    status = RunNodes();
    if (status != hipSuccess) {
      LogError("Failed to launch nodes!");
      this->release();
      return status;
    }
  }

  if (DEBUG_HIP_GRAPH_DOT_PRINT == 2 && !graph_dumped_) {
    graph_dumped_ = true;
    std::string filename =
        "graph_" + std::to_string(amd::Os::getProcessId()) + "_dot_print_launch_1";
    hipError_t dot_status = ihipGraphDebugDotPrint(this, filename.c_str(), 0);
    if (dot_status == hipSuccess) {
      ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_CODE, "[hipGraph] graph dump:%s", filename.c_str());
    }
  }

  // Enqueue a lightweight marker to carry the launch-complete callback.
  auto* marker = new amd::Marker(*launch_stream, kMarkerDisableFlush, {});
  marker->setCommandEntryScope(amd::Device::kCacheStateIgnore);
  amd::Event& event = marker->event();
  constexpr bool kBlocking = false;
  auto* cleanup = new GraphLaunchCleanup();
  cleanup->exec = this;
  cleanup->device = g_devices[launch_stream->DeviceId()]->devices()[0];
  if (!event.setCallback(CL_COMPLETE, GraphExecBase::OnLaunchComplete, cleanup, kBlocking)) {
    launch_stream->finish();
    GraphExecBase::OnLaunchComplete(nullptr, CL_COMPLETE, cleanup);
    marker->release();
    return hipErrorInvalidHandle;
  }
  marker->enqueue();
  marker->release();
  return status;
}

// ================================================================================================
hipError_t GraphExecSegmented::Init() {
  hipError_t status = hipSuccess;

  // Schedule nodes into segments for batch execution
  status = ScheduleNodesIntoBatches();
  if (status != hipSuccess) {
    return status;
  }

  // Allocate kernel argument manager for packet capture
  if (kernArgManager_ == nullptr) {
    SetKernelArgManager(new GraphKernelArgManager());
  }

  // captureDeviceId_ is set inside FindStreamsReqPerDevForSegments() from
  // max_streams_dev_ once all segments are analysed. Must run unconditionally
  // so that captureDeviceId_ is valid before CaptureAQLPackets/BuildSyncPlan.
  status = FindStreamsReqPerDevForSegments();
  if (status != hipSuccess) {
    return status;
  }

  // create extra stream to avoid queue collision with the default execution stream
  if (max_streams_ >= 1) {
    // Cap per-device stream counts to the hardware queue limit.
    // SelectStreamAssignment() reads max_streams_dev_ to assign segment stream ids,
    // so both must see the capped values.
    for (auto& [dev_id, count] : max_streams_dev_) {
      count = std::min(count, static_cast<int>(DEBUG_HIP_FORCE_GRAPH_QUEUES));
    }
  }

  // Select and apply stream assignment before packet capture so that BuildSyncPlan
  // (called inside CaptureAQLPackets) can see each segment's stream_id and
  // skip same-stream dependency barriers.
  SelectStreamAssignment();

  // For graph nodes capture AQL packets to dispatch them directly during graph launch.
  // BuildSyncPlan (inside CaptureAndFormPacketsForGraph) runs the barrier-ROI collapse
  // pass, which may fold the graph onto a single stream per device.
  status = CaptureAQLPackets();
  if (status != hipSuccess) {
    return status;
  }

  // Create parallel streams now (still at instantiate time, never lazily at launch),
  // sized to the final post-collapse assignment: one stream per device when the
  // collapse pass fired, otherwise the capped multi-stream counts.
  if (collapsed_to_single_stream_) {
    for (auto& [dev_id, count] : max_streams_dev_) {
      count = 1;
    }
  }
  uint32_t total_streams = 0;
  for (auto const& [dev_id, count] : max_streams_dev_) {
    total_streams += static_cast<uint32_t>(std::max(count, 0));
  }
  ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_CODE,
          "[hipGraph] Init: %zu device(s), %u total stream(s) (per-device cap: %u)",
          max_streams_dev_.size(), total_streams, DEBUG_HIP_FORCE_GRAPH_QUEUES);
  for (auto const& [dev_id, num_streams] : max_streams_dev_) {
    if (num_streams > 0) {
      status = CreateStreams(num_streams, dev_id);
      if (status != hipSuccess) {
        return status;
      }
    }
  }

  return status;
}

//! Chunk size to add to kern arg pool
constexpr uint32_t kKernArgChunkSize = 128 * Ki;
// ================================================================================================
void GraphExecSegmented::GetKernelArgSizeForGraph(std::unordered_map<int, size_t>& kernArgSizeForGraph) {
  // Calculate the kernel argument size required for all graph kernel nodes
  // when GPU packet capture is enabled

  if (!segments_.empty()) {
    for (const auto& segment : segments_) {
      // Handle child graph segments - skip node iteration, process recursively
      if (segment.child_graph_ptr != nullptr) {
        auto childGraphExec = dynamic_cast<GraphExecSegmented*>(segment.child_graph_ptr);
        if (childGraphExec != nullptr) {
          // Child graphs share the same kernel arg manager as parent
          if (childGraphExec->GetKernelArgManager() == nullptr) {
            auto kernArgMgr = GetKernelArgManager();
            if (kernArgMgr != nullptr) {
              kernArgMgr->retain();  // Increment ref count for child's reference
              childGraphExec->SetKernelArgManager(kernArgMgr);
            }
          }
          childGraphExec->GetKernelArgSizeForGraph(kernArgSizeForGraph);
        }
        continue;  // Skip processing nodes in this segment
      }

      // Process regular nodes in this segment
      for (hip::GraphNode* node : segment.nodes) {
        if (node->GraphCaptureEnabled()) {
          // Accumulate the kernel argument size for each device
          kernArgSizeForGraph[node->dev_id_] += node->GetKerArgSize();
        }
      }
    }
  }
}
// ================================================================================================
// Enable or disable a graph node's packets in the batch
// Simply updates the enabled state and count of disabled nodes
void GraphExecSegmented::PacketBatch::setEnabled(GraphNode* node, bool enabled) {
  auto it = nodeToRangeIndex.find(node);
  if (it == nodeToRangeIndex.end()) {
    return;
  }
  NodeRange& range = nodeRanges[it->second];
  // Early return if state hasn't changed
  if (range.enabled == enabled) {
    return;
  }
  // Update counter based on state change
  if (enabled) {
    // Node being enabled: decrement counter
    // Defensive check to prevent underflow
    if (disabledNodeCount > 0) {
      disabledNodeCount--;
    }
  } else {
    // Node being disabled: increment counter
    disabledNodeCount++;
  }
  range.enabled = enabled;
  filteredCacheValid = false;
}

// ================================================================================================
// Rebuild cached filtered lists of enabled packets.
// Barrier packets (prepended/appended by BuildSyncPlan) live in dispatchPackets
// but are not tracked in nodeRanges.  A linear scan with a per-index enabled
// bitmap preserves them while filtering out disabled node packets.
// ================================================================================================
void GraphExecSegmented::PacketBatch::rebuildFilteredLists(
    std::vector<amd::Device::HwEventPatch>& patch_list) {
  if (filteredCacheValid) {
    return;
  }

  // Default every packet to enabled; mark disabled-node packets false.
  std::vector<bool> packetEnabled(dispatchPackets.size(), true);
  for (const auto& range : nodeRanges) {
    if (!range.enabled) {
      for (size_t j = 0; j < range.packetCount; ++j) {
        packetEnabled[range.startIndex + j] = false;
      }
    }
  }

  enabledPackets.clear();
  enabledKernelNames.clear();
  filteredFlatPacketData.clear();
  filteredValidPacketFullHeaders.clear();
  filteredFlatMetadataData.clear();

  enabledPackets.reserve(dispatchPackets.size());
  enabledKernelNames.reserve(dispatchPackets.size());
  filteredFlatPacketData.reserve(dispatchPackets.size() * kAqlPktSize);
  filteredValidPacketFullHeaders.reserve(dispatchPackets.size());
  filteredFlatMetadataData.reserve(dispatchPackets.size() * kMetadataPktSize);

  const bool hasMetadata = !dispatchMetadataPackets.empty();

  // packet pointer -> index in the filtered flat buffer, built during the
  // single pass below so patch_list resolution is O(patches) not O(p*n).
  std::unordered_map<const void*, size_t> packetToFilteredIndex;

  // Packets in THIS batch whose owning node is disabled (filtered out below). A
  // completion-signal patch pinned to one of these must be relocated, otherwise
  // the segment never emits its signal and any consumer waiting on it deadlocks.
  std::unordered_set<const void*> disabledBatchPackets;

  // Packets in this batch that already carry a completion signal, so a relocation cannot
  // land on top of one. Seeded from the patch list rather than assumed empty: with interior
  // producers an ordinary dispatch packet can be a completion-signal carrier.
  std::unordered_set<const uint8_t*> claimedForCompletion;
  for (const auto& patch : patch_list) {
    if (patch.dep_slot == amd::Device::HwEventPatch::kCompletionSignal) {
      claimedForCompletion.insert(patch.packet);
    }
  }

  for (size_t i = 0; i < dispatchPackets.size(); ++i) {
    if (packetEnabled[i]) {
      size_t filteredIdx = enabledPackets.size();
      enabledPackets.push_back(dispatchPackets[i]);
      enabledKernelNames.push_back(dispatchKernelNames[i]);
      // appendPacketToFlatBuffer also appends the index-aligned metadata slot
      // (zero-filled when this index has no metadata packet). Empty slots
      // (barriers / uncaptured dispatches) are then stamped
      // HSA_PACKET_TYPE_INVALID so the CP prefetch engine skips them — a zeroed
      // slot would be type 0 (VENDOR_SPECIFIC), which the CP could process.
      const uint8_t* metadata_raw =
          (hasMetadata && i < dispatchMetadataPackets.size()) ? dispatchMetadataPackets[i]
                                                              : nullptr;
      appendPacketToFlatBuffer(dispatchPackets[i], metadata_raw, filteredFlatPacketData,
                               filteredValidPacketFullHeaders, filteredFlatMetadataData);
      if (hasMetadata && metadata_raw == nullptr) {
        invalidateMetadataSlot(filteredFlatMetadataData.data() +
                               filteredFlatMetadataData.size() - kMetadataPktSize);
      }

      packetToFilteredIndex[dispatchPackets[i]] = filteredIdx;
    } else {
      disabledBatchPackets.insert(dispatchPackets[i]);
    }
  }

  // Re-point flat_packet pointers in patch_list into filteredFlatPacketData.
  for (auto& patch : patch_list) {
    auto it = packetToFilteredIndex.find(patch.packet);
    if (it != packetToFilteredIndex.end()) {
      patch.flat_packet =
          filteredFlatPacketData.data() + it->second * kAqlPktSize;
      continue;
    }

    // The patch's owning packet is not in the filtered buffer. Only touch patches
    // this batch owns whose packet was disabled; patches for other batches are
    // resolved when those batches rebuild.
    if (disabledBatchPackets.find(patch.packet) == disabledBatchPackets.end()) {
      continue;
    }

    // Only completion signals are relocatable here.
    if (patch.dep_slot != amd::Device::HwEventPatch::kCompletionSignal) {
      continue;
    }

    // Relocate the signal to the last still-enabled packet of this batch that is not
    // ALREADY carrying a completion signal. Interior producers put completion signals on
    // ordinary dispatch packets, so the naive "last enabled packet" can already be spoken
    // for, and writing a second signal there would silently drop the first.
    size_t relocate_idx = enabledPackets.size();
    for (size_t k = enabledPackets.size(); k-- > 0;) {
      if (claimedForCompletion.find(enabledPackets[k]) == claimedForCompletion.end()) {
        relocate_idx = k;
        break;
      }
    }
    if (relocate_idx < enabledPackets.size()) {
      patch.flat_packet = filteredFlatPacketData.data() + relocate_idx * kAqlPktSize;
      claimedForCompletion.insert(enabledPackets[relocate_idx]);
    } else if (fallbackBarrier != nullptr) {
      // Every node packet in this batch is disabled: no packet remains to host
      // the signal. Splice the reserved standalone barrier into the *filtered*
      // buffer only — dispatchPackets/flatPacketData stay untouched.
      const size_t fallback_idx = enabledPackets.size();
      enabledPackets.push_back(fallbackBarrier);
      enabledKernelNames.push_back(nullptr);
      appendPacketToFlatBuffer(fallbackBarrier, nullptr, filteredFlatPacketData,
                               filteredValidPacketFullHeaders, filteredFlatMetadataData);
      if (hasMetadata) {
        invalidateMetadataSlot(filteredFlatMetadataData.data() +
                               filteredFlatMetadataData.size() - kMetadataPktSize);
      }
      patch.flat_packet =
          filteredFlatPacketData.data() + fallback_idx * kAqlPktSize;
      claimedForCompletion.insert(fallbackBarrier);
    } else {
      // No carrier left at all. Say so: the alternative is a consumer that waits forever
      // with nothing in the log.
      ClPrint(amd::LOG_ERROR, amd::LOG_CODE,
              "[hipGraph] COMPLETION SIGNAL DROPPED: batch has no packet left to carry it "
              "after node disable; consumers of this signal will deadlock");
    }
  }

  filteredCacheValid = true;
}

// ================================================================================================
// Restore flat_packet pointers in patch_list back to flatPacketData.
// Called when all nodes are re-enabled (disabledNodeCount == 0) so that
// ApplyHwEventPatches writes into the buffer the dispatch path will use.
// ================================================================================================
void GraphExecSegmented::PacketBatch::restorePatchListPointers(
    std::vector<amd::Device::HwEventPatch>& patch_list) {
  for (auto& patch : patch_list) {
    for (size_t i = 0; i < dispatchPackets.size(); ++i) {
      if (patch.packet == dispatchPackets[i]) {
        patch.flat_packet = flatPacketData.data() + i * kAqlPktSize;
        break;
      }
    }
  }
}

// ================================================================================================
hipError_t GraphExecSegmented::CaptureAndFormPacketsForGraph() {
  // Fixme: Only single stream child graph nodes are supported.
  hipError_t status = hipSuccess;

  // Clear previous batches
  segmentBatches_.clear();

  // Process nodes from segments
  for (const auto& segment : segments_) {
    // Child-graph segments: create a SegmentBatch with leading + trailing empty batches
    // so BuildSyncPlan can prepend dep barriers and append a completion barrier
    if (segment.child_graph_ptr != nullptr) {
      auto [it, inserted] = segmentBatches_.emplace(segment.id, segment.id);
      auto& childSegBatch = it->second;
      childSegBatch.node_capture_status.resize(segment.nodes.size(), false);
      childSegBatch.has_uncaptured_nodes = true;
      childSegBatch.packet_batches.emplace_back();  // leading: dep barriers
      childSegBatch.packet_batches.emplace_back();  // trailing: completion barrier
      continue;
    }

    // Always create a SegmentBatch for every non-child-graph segment
    auto [it, inserted] = segmentBatches_.emplace(segment.id, segment.id);
    // Initialize node_capture_status for this segment
    auto& currentSegBatch = it->second;
    currentSegBatch.node_capture_status.resize(segment.nodes.size(), false);

    bool first_node_is_uncaptured = !segment.nodes.empty() &&
                                    !segment.nodes[0]->GraphCaptureEnabled();

    // Leading empty batch: gives BuildSyncPlan a slot to prepend the cross-dep
    // BARRIER_AND so it physically precedes the uncaptured node's GPU commands.
    if (first_node_is_uncaptured) {
      currentSegBatch.packet_batches.emplace_back();
    }

    for (size_t i = 0; i < segment.nodes.size(); ++i) {
      auto& node = segment.nodes[i];

      // Check if kernel node requires hidden heap and set it for the entire graph
      if (node->GetType() == hipGraphNodeTypeKernel) {
        static bool initialized = false;
        if (!initialized && reinterpret_cast<hip::GraphKernelNode*>(node)->HasHiddenHeap()) {
          SetHiddenHeap();
          initialized = true;
        }
      }

      if (node->GraphCaptureEnabled()) {
        // Start of a new batch
        PacketBatch newBatch;

        // Collect packets from consecutive captured nodes
        size_t j = i;
        while (j < segment.nodes.size() && segment.nodes[j]->GraphCaptureEnabled()) {
          auto& currentNode = segment.nodes[j];
          // Empty nodes are pure dependency points — no GPU commands or markers.
          // Cross-stream ordering is handled by BuildSyncPlan barrier packets.
          if (currentNode->GetType() == hipGraphNodeTypeEmpty) {
            const size_t rangeIndex = newBatch.nodeRanges.size();
            newBatch.nodeRanges.push_back({newBatch.dispatchPackets.size(), 0, true});
            newBatch.nodeToRangeIndex[currentNode] = rangeIndex;
            currentSegBatch.node_capture_status[j] = true;
            ++j;
            continue;
          }
          // Capture packets for this node
          std::vector<uint8_t*> nodePackets;
          std::vector<const std::string*> nodeKernelNames;
          std::vector<uint8_t*> nodeMetadataPackets;
          status = currentNode->CaptureAndFormPacket(GetKernelArgManager(), &nodePackets,
                                                     &nodeKernelNames, &nodeMetadataPackets);

          if (status != hipSuccess || nodePackets.empty()) {
            LogError("Packet capture failed");
            return status;
          }

          // Create NodeRange for this node
          // RangeIndex is 0 at the start
          const size_t rangeIndex = newBatch.nodeRanges.size();
          const size_t startIndex = newBatch.dispatchPackets.size();
          const size_t packetCount = nodePackets.size();

          // Reserve space to avoid reallocations during insertion
          newBatch.dispatchPackets.reserve(startIndex + packetCount);
          newBatch.dispatchKernelNames.reserve(startIndex + packetCount);
          newBatch.dispatchMetadataPackets.reserve(startIndex + packetCount);

          // Add to dispatch lists (initially all enabled)
          newBatch.dispatchPackets.insert(newBatch.dispatchPackets.end(), nodePackets.begin(),
                                          nodePackets.end());
          newBatch.dispatchKernelNames.insert(newBatch.dispatchKernelNames.end(),
                                              nodeKernelNames.begin(), nodeKernelNames.end());
          newBatch.dispatchMetadataPackets.insert(newBatch.dispatchMetadataPackets.end(),
                                                  nodeMetadataPackets.begin(),
                                                  nodeMetadataPackets.end());

          // Store node mapping with range info
          newBatch.nodeRanges.push_back({startIndex, packetCount, true});
          newBatch.nodeToRangeIndex[currentNode] = rangeIndex;

          // Mark this node as successfully captured
          currentSegBatch.node_capture_status[j] = true;
          ++j;
        }

        // Add the batch if it contains packets or captured zero-packet nodes (e.g. EMPTY).
        if (!newBatch.dispatchPackets.empty() || !newBatch.nodeRanges.empty()) {
          currentSegBatch.packet_batches.push_back(std::move(newBatch));
        }
        i = j - 1;  // for-loop will ++i to j
      } else {
        // Non-capturable node
        currentSegBatch.has_uncaptured_nodes = true;
        currentSegBatch.node_capture_status[i] = false;
      }
    }

    // Trailing empty batch: separate slot for the completion barrier so it
    // cannot fire before the uncaptured last node's commands finish.
    bool last_node_uncaptured = currentSegBatch.has_uncaptured_nodes &&
        !segment.nodes.empty() && !currentSegBatch.node_capture_status.back();
    if (last_node_uncaptured) {
      currentSegBatch.packet_batches.emplace_back();
    }
  }

  // Recursively process child graphs to capture their packets
  for (const auto& segment : segments_) {
    if (segment.child_graph_ptr != nullptr) {
      auto childGraphExec = dynamic_cast<GraphExecSegmented*>(segment.child_graph_ptr);
      if (childGraphExec != nullptr) {
        if (childGraphExec->captureDeviceId_ == -1) {
          childGraphExec->captureDeviceId_ = captureDeviceId_;
        }

        // Child graphs share the same kernel arg manager as parent
        // This is critical for packet capture to work correctly
        if (childGraphExec->GetKernelArgManager() == nullptr) {
          auto kernArgMgr = GetKernelArgManager();
          if (kernArgMgr != nullptr) {
            kernArgMgr->retain();  // Increment ref count for child's reference
            childGraphExec->SetKernelArgManager(kernArgMgr);
          }
        }

        status = childGraphExec->CaptureAndFormPacketsForGraph();
        if (status != hipSuccess) {
          ClPrint(amd::LOG_ERROR, amd::LOG_CODE,
                  "[hipGraph] Child graph packet capture failed for child graph in segment, "
                  "status=%d",
                  status);
          return status;
        }
      }
    }
  }

  // Build sync plan now that all segment batches are populated --
  // prepends barrier packets and generates the patch list
  BuildSyncPlan();

  // Build flat buffers once now that all dispatchPackets are finalized
  // (capture populated them, BuildSyncPlan may have prepended/appended barriers).
  // Also build a map from dispatchPacket pointer -> flat buffer pointer so
  // ApplyHwEventPatches can patch flatPacketData directly at launch time.
  std::unordered_map<const uint8_t*, uint8_t*> pktToFlat;
  for (auto& [seg_id, segBatch] : segmentBatches_) {
    for (auto& batch : segBatch.packet_batches) {
      if (!batch.dispatchPackets.empty()) {
        batch.rebuildFlatBuffer();
        for (size_t i = 0; i < batch.dispatchPackets.size(); ++i) {
          pktToFlat[batch.dispatchPackets[i]] =
              batch.flatPacketData.data() + i * PacketBatch::kAqlPktSize;
        }
      }
    }
  }

  // Resolve flat_packet pointers for all HwEventPatches.
  // ⛔ A MISS used to leave flat_packet at whatever it already held -- nullptr for every
  // patch -- and ApplyHwEventPatches writes THROUGH it, so the failure surfaces as an
  // aperture violation at launch rather than at the miss. Count them and drop them, so a
  // miss loses ordering (detectable) instead of faulting the GPU (not attributable).
  size_t unresolved = 0;
  for (auto& patch : sync_plan_.patch_list) {
    auto it = pktToFlat.find(patch.packet);
    if (it != pktToFlat.end()) {
      patch.flat_packet = it->second;
    } else {
      ++unresolved;
      patch.flat_packet = nullptr;
    }
  }
  if (unresolved != 0) {
    ClPrint(amd::LOG_ERROR, amd::LOG_CODE,
            "[hipGraph] PATCH UNRESOLVED: %zu of %zu HwEventPatches have no flat packet -- "
            "dropped rather than written through null; their ordering is LOST",
            unresolved, sync_plan_.patch_list.size());
    sync_plan_.patch_list.erase(
        std::remove_if(sync_plan_.patch_list.begin(), sync_plan_.patch_list.end(),
                       [](const amd::Device::HwEventPatch& p) { return p.flat_packet == nullptr; }),
        sync_plan_.patch_list.end());
  }
  ClPrint(amd::LOG_INFO, amd::LOG_CODE,
          "[hipGraph] PatchResolve: total=%zu unresolved=%zu",
          sync_plan_.patch_list.size(), unresolved);

  return status;
}

// ================================================================================================
hipError_t GraphExecSegmented::CaptureAQLPackets() {
  hipError_t status = hipSuccess;

  // Create a map to track kernel argument sizes for each device
  std::unordered_map<int, size_t> kernArgSizeForGraph;
  // Reserve space for all available devices and Initialize to 0
  kernArgSizeForGraph.reserve(g_devices.size());
  for (int devId = 0; devId < g_devices.size(); devId++) {
    kernArgSizeForGraph[devId] = 0;
  }
  GetKernelArgSizeForGraph(kernArgSizeForGraph);

  // Allocate kernel argument pools on respective devices with extra space for updates
  for (const auto& deviceKernArgPair : kernArgSizeForGraph) {
    const int deviceId = deviceKernArgPair.first;
    const size_t kernArgSize = deviceKernArgPair.second;

    if (kernArgSize == 0) {
      continue;
    }

    const size_t totalPoolSize = kernArgSize + kKernArgChunkSize;
    if (!kernArgManager_->AllocGraphKernargPool(totalPoolSize, g_devices[deviceId]->devices()[0])) {
      ClPrint(amd::LOG_ERROR, amd::LOG_CODE,
              "[hipGraph] Failed to allocate kernel argument pool of size %zu for device %d",
              totalPoolSize, deviceId);
    return hipErrorMemoryAllocation;
    }
  }

  status = CaptureAndFormPacketsForGraph();
  if (status != hipSuccess) {
    return status;
  }

  kernArgManager_->ReadBackOrFlush();
  return hipSuccess;;
}

// ================================================================================================
hipError_t GraphExecSegmented::UpdateAQLPacket(hip::GraphNode* node) {
  if (!node->GraphCaptureEnabled()) {
    return hipSuccess;
  }
  // Todo: Add batching support for multi-device linear graph
  // Use node_to_segment_id_ for O(1) segment lookup
  auto segIdIt = node_to_segment_id_.find(node);
  if (segIdIt == node_to_segment_id_.end()) {
    return hipSuccess;  // Node not in any segment
  }

  int segmentId = segIdIt->second;

  // Find the segment batch for this segment ID using O(1) map lookup
  auto segBatchIt = segmentBatches_.find(segmentId);
  if (segBatchIt == segmentBatches_.end()) {
    return hipSuccess;  // Segment not found
  }

  auto& segBatch = segBatchIt->second;

  // Search only within this segment's packet batches
  for (auto& packetBatch : segBatch.packet_batches) {
    auto it = packetBatch.nodeToRangeIndex.find(node);
    if (it != packetBatch.nodeToRangeIndex.end()) {
      // Found the batch containing this node - update packets
      PacketBatch::NodeRange& range = packetBatch.nodeRanges[it->second];

      // Capture new packets for this node
      std::vector<uint8_t*> newPackets;
      std::vector<const std::string*> newKernelNames;
      std::vector<uint8_t*> newMetadataPackets;
      // A disabled node's CreateCommand returns early without emitting any
      // command, so CaptureAndFormPacket would yield zero packets and the
      // packet-count-change path below would delete the node's dispatch slot
      // entirely -- leaving nothing to run once it is re-enabled. Force the
      // node enabled just for the capture so the updated packet materializes,
      // then restore the disabled state.
      const unsigned int saved_enabled_state = node->GetEnabled();
      if (saved_enabled_state == 0) {
        node->SetEnabled(1);
      }
      hipError_t status = node->CaptureAndFormPacket(kernArgManager_, &newPackets,
                                                                      &newKernelNames,
                                                                      &newMetadataPackets);
      node->SetEnabled(saved_enabled_state);
      if (status != hipSuccess) {
        return status;
      }
      // Number of packets per node can change
      const size_t oldPacketCount = range.packetCount;
      const size_t newPacketCount = newPackets.size();

      if (newPacketCount != oldPacketCount) {
        const size_t rangeIdx = it->second;
        const int64_t packetDelta =
            static_cast<int64_t>(newPacketCount) - static_cast<int64_t>(oldPacketCount);

        ClPrint(
            amd::LOG_DETAIL_DEBUG, amd::LOG_CODE,
            "[hipGraph] Packet count change for node (type=%d): %zu -> %zu packets (delta=%ld)",
            node->GetType(), oldPacketCount, newPacketCount, packetDelta);

        if (packetDelta > 0) {
          // Insert additional packet slots at the end of this node's range
          const size_t insertPos = range.startIndex + oldPacketCount;
          packetBatch.dispatchPackets.insert(packetBatch.dispatchPackets.begin() + insertPos,
                                             static_cast<size_t>(packetDelta), nullptr);
          packetBatch.dispatchKernelNames.insert(
              packetBatch.dispatchKernelNames.begin() + insertPos,
              static_cast<size_t>(packetDelta), nullptr);
          if (packetBatch.dispatchMetadataPackets.size() >= insertPos) {
            packetBatch.dispatchMetadataPackets.insert(
                packetBatch.dispatchMetadataPackets.begin() + insertPos,
                static_cast<size_t>(packetDelta), nullptr);
          }
        } else {
          // Negative packetDelta, remove excess packet slots from the end of this node's range
          const size_t removePos = range.startIndex + newPacketCount;
          const size_t removeCount = oldPacketCount - newPacketCount;

          // Validate bounds before erasing
          if (removePos + removeCount > packetBatch.dispatchPackets.size()) {
            ClPrint(amd::LOG_ERROR, amd::LOG_CODE,
                    "[hipGraph] Invalid packet removal bounds: pos=%zu, count=%zu, size=%zu",
                    removePos, removeCount, packetBatch.dispatchPackets.size());
            return hipErrorInvalidValue;
          }

          packetBatch.dispatchPackets.erase(
              packetBatch.dispatchPackets.begin() + removePos,
              packetBatch.dispatchPackets.begin() + removePos + removeCount);
          packetBatch.dispatchKernelNames.erase(
              packetBatch.dispatchKernelNames.begin() + removePos,
              packetBatch.dispatchKernelNames.begin() + removePos + removeCount);
          if (packetBatch.dispatchMetadataPackets.size() >= removePos + removeCount) {
            packetBatch.dispatchMetadataPackets.erase(
                packetBatch.dispatchMetadataPackets.begin() + removePos,
                packetBatch.dispatchMetadataPackets.begin() + removePos + removeCount);
          }
        }

        // Update this node's packet count and adjust startIndex for all subsequent nodes
        range.packetCount = newPacketCount;
        for (size_t i = rangeIdx + 1; i < packetBatch.nodeRanges.size(); ++i) {
          packetBatch.nodeRanges[i].startIndex = static_cast<size_t>(
              static_cast<int64_t>(packetBatch.nodeRanges[i].startIndex) + packetDelta);
        }
      }

      // Metadata-prefetch packets are kept parallel to dispatchPackets; the vector is
      // empty when the gfx1250 prefetch path is inactive.
      const bool hasMetadata = !packetBatch.dispatchMetadataPackets.empty();

      // Update dispatch packets (always update regardless of enabled state)
      // The enabled/disabled check happens during dispatch, not here
      for (size_t i = 0; i < range.packetCount && i < newPackets.size(); ++i) {
        size_t packetIndex = range.startIndex + i;
        uint8_t* oldPkt = packetBatch.dispatchPackets[packetIndex];
        uint8_t* newPkt = newPackets[i];
        packetBatch.dispatchPackets[packetIndex] = newPkt;
        packetBatch.dispatchKernelNames[packetIndex] = newKernelNames[i];
        if (hasMetadata) {
          packetBatch.dispatchMetadataPackets[packetIndex] =
              (i < newMetadataPackets.size()) ? newMetadataPackets[i] : nullptr;
        }

        // Update SyncPlan patch list to point to the new packet
        // ApplyHwEventPatches patches the correct packet at launch time.
        if (oldPkt != newPkt) {
          for (auto& patch : sync_plan_.patch_list) {
            if (patch.packet == oldPkt) {
              patch.packet = newPkt;
            }
          }
        }
      }
      // Rebuild the flat buffer immediately so the next dispatch uses updated packets.
      // The flat buffer always represents the full packet sequence; the dispatch path
      // independently skips it when any nodes are disabled (disabledNodeCount != 0).
      packetBatch.rebuildFlatBuffer();

      // Refresh flat_packet pointers in the patch list since rebuildFlatBuffer
      // reallocated flatPacketData, invalidating previous flat_packet pointers.
      for (auto& patch : sync_plan_.patch_list) {
        for (size_t pi = 0; pi < packetBatch.dispatchPackets.size(); ++pi) {
          if (patch.packet == packetBatch.dispatchPackets[pi]) {
            patch.flat_packet = packetBatch.flatPacketData.data() + pi * PacketBatch::kAqlPktSize;
            break;
          }
        }
      }
      return hipSuccess;
    }
  }
  return hipSuccess;  // Node not in any batch
}

// ================================================================================================
// Append one 64-byte AQL packet to a flat buffer: copies the body, saves the original full_header
// and invalidates the header.
void GraphExecSegmented::PacketBatch::appendPacketToFlatBuffer(const uint8_t* pkt_raw,
                                                      const uint8_t* metadata_raw,
                                                      amd::AlignedVector64<uint8_t>& flatData,
                                                      std::vector<uint32_t>& fullHeaders,
                                                      std::vector<uint8_t>& flatMetadata) {
  static constexpr size_t kSigOff = 56;
  const size_t baseOff = flatData.size();
  flatData.insert(flatData.end(), pkt_raw, pkt_raw + kAqlPktSize);
  uint8_t* dst = flatData.data() + baseOff;
  uint32_t fullHeader = 0;
  memcpy(&fullHeader, pkt_raw, sizeof(fullHeader));
  fullHeaders.push_back(fullHeader);
  // Set header to HSA_PACKET_TYPE_INVALID (type=1) so the GPU CP skips this
  // packet until the valid header is committed with release semantics during
  // dispatch. Using type=0 (VENDOR_SPECIFIC) would be a processable packet type
  // that the CP could attempt to execute with incomplete body data.
  static constexpr uint16_t kInvalidAqlHeader = 1;
       //HSA_PACKET_TYPE_INVALID << HSA_PACKET_HEADER_TYPE;
  memcpy(dst, &kInvalidAqlHeader, sizeof(kInvalidAqlHeader));
  // Zero completion signal; ApplyHwEventPatches re-patches it directly via flat_packet pointers.
  memset(dst + kSigOff, 0, sizeof(uint64_t));

  // Append the matching metadata-prefetch packet so flatMetadata stays index-
  // aligned with flatData. A nullptr |metadata_raw| yields a zeroed slot.
  const size_t metaOff = flatMetadata.size();
  flatMetadata.insert(flatMetadata.end(), kMetadataPktSize, 0);
  if (metadata_raw != nullptr) {
    memcpy(flatMetadata.data() + metaOff, metadata_raw, kMetadataPktSize);
  }
}

// ================================================================================================
// Publish an empty metadata slot as HSA_PACKET_TYPE_INVALID. The four metadata packet headers
// live at offsets {0, 64, 128, 192} within the 256-byte AqlMetadataPrefetchPacket layout, and
// the low byte of each is the packet type.
void GraphExecSegmented::PacketBatch::invalidateMetadataSlot(uint8_t* slot) {
  static constexpr size_t kHdrOff[4] = {0, 64, 128, 192};
  static constexpr uint32_t kInvalidMetadataHeader = 1;  // HSA_PACKET_TYPE_INVALID
  for (size_t h = 0; h < 4; ++h) {
    std::memcpy(slot + kHdrOff[h], &kInvalidMetadataHeader, sizeof(kInvalidMetadataHeader));
  }
}

// ================================================================================================
// Rebuild the flat packet buffer from the current dispatchPackets contents.
void GraphExecSegmented::PacketBatch::rebuildFlatBuffer() {
  const size_t n = dispatchPackets.size();
  flatPacketData.clear();
  validPacketFullHeaders.clear();
  flatMetadataData.clear();
  filteredCacheValid = false;
  flatPacketData.reserve(n * kAqlPktSize);
  validPacketFullHeaders.reserve(n);
  flatMetadataData.reserve(n * kMetadataPktSize);
  for (size_t i = 0; i < n; ++i) {
    const uint8_t* metadata_raw =
        (i < dispatchMetadataPackets.size()) ? dispatchMetadataPackets[i] : nullptr;
    appendPacketToFlatBuffer(dispatchPackets[i], metadata_raw, flatPacketData,
                             validPacketFullHeaders, flatMetadataData);
  }
  // Build flat metadata buffer (kMetadataPktSize per slot).
  // Slots without captured metadata (barriers, uncaptured dispatches) are published as
  // HSA_PACKET_TYPE_INVALID so the CP prefetch engine skips them.
  if (!dispatchMetadataPackets.empty()) {
    flatMetadataData.resize(n * kMetadataPktSize, 0);
    for (size_t i = 0; i < n; ++i) {
      uint8_t* slot = flatMetadataData.data() + i * kMetadataPktSize;
      if (i < dispatchMetadataPackets.size() && dispatchMetadataPackets[i] != nullptr) {
        std::memcpy(slot, dispatchMetadataPackets[i], kMetadataPktSize);
      } else {
        invalidateMetadataSlot(slot);
      }
    }
  }
}

// ================================================================================================
hipError_t GraphExecSegmented::UpdatePacketBatchesForNodeEnableDisable(hip::GraphNode* node,
                                                              bool isEnabled) {
  if (!node->GraphCaptureEnabled()) {
    // Only handle single stream case with captured nodes
    return hipSuccess;
  }

  // Use node_to_segment_id_ for O(1) segment lookup
  auto segIdIt = node_to_segment_id_.find(node);
  if (segIdIt == node_to_segment_id_.end()) {
    return hipSuccess; // Node not in any segment
  }

  int segmentId = segIdIt->second;

  // Find the segment batch for this segment ID using O(1) map lookup
  auto segBatchIt = segmentBatches_.find(segmentId);
  if (segBatchIt == segmentBatches_.end()) {
    return hipSuccess; // Segment not found
  }

  auto& segBatch = segBatchIt->second;

  // Search only within this segment's packet batches
  for (auto& packetBatch : segBatch.packet_batches) {
    auto it = packetBatch.nodeToRangeIndex.find(node);
    if (it != packetBatch.nodeToRangeIndex.end()) {
      // Found the batch containing this node - update enabled state
      packetBatch.setEnabled(node, isEnabled);
      if (packetBatch.disabledNodeCount > 0) {
        // Eagerly rebuild filtered lists and re-resolve patch_list flat_packet
        // pointers so the launch path doesn't need to scan all segment batches.
        packetBatch.rebuildFilteredLists(sync_plan_.patch_list);
      } else {
        // All nodes re-enabled: restore flat_packet pointers back to
        // flatPacketData so ApplyHwEventPatches patches the correct buffer.
        packetBatch.restorePatchListPointers(sync_plan_.patch_list);
      }
      return hipSuccess;
    }
  }
  return hipSuccess;
}

void GraphExecBase::OnLaunchComplete(cl_event event, cl_int command_exec_status, void* user_data) {
  auto* cleanup = reinterpret_cast<GraphLaunchCleanup*>(user_data);
  GraphExecBase* execBase = cleanup->exec;
  // Re-arm and recycle the launch's signals while the GraphExecBase (and thus its
  // signal pool) is still alive, then drop the launch's reference.
  execBase->RecycleLaunchSignals(cleanup->device, cleanup->signal_set);
  delete cleanup;
  execBase->release();
}

// ================================================================================================
amd::Command* GraphExecSegmented::EnqueueSegmentedGraph(hip::Stream* launch_stream,
                                               const std::vector<hip::Stream*>& streams,
                                               hipError_t* out_status,
                                               std::vector<void*>* out_signal_set) {
  hipError_t status = hipSuccess;
  if (out_status != nullptr) {
    *out_status = hipSuccess;
  }

  auto* device = g_devices[launch_stream->DeviceId()]->devices()[0];

  // Top-level launches recycle signals through the per-graph pool; the legacy
  // recursive child-graph path (out_signal_set == nullptr) creates them locally
  // and lets the AccumulateCommand destructor destroy them.
  const bool recycle = (out_signal_set != nullptr);

  std::vector<void*> segment_hw_events;
  if (sync_plan_.num_hw_events > 0) {
    const bool ok = recycle
        ? signalManager_->AcquireSet(device, sync_plan_.num_hw_events, segment_hw_events)
        : device->CreateHwEvents(sync_plan_.num_hw_events, segment_hw_events);
    if (!ok) {
      if (out_status != nullptr) {
        *out_status = hipErrorOutOfMemory;
      }
      return nullptr;
    }
  }

  // Resolve a segment's assigned hip::Stream* from its pre-computed stream_id.
  // streams is the collision-handled streams_ vector built by UpdateStreams.
  auto resolveSegmentStream = [&](const Segment& seg) -> hip::Stream* {
    if (!streams.empty()) {
      return streams[static_cast<size_t>(seg.stream_id) % streams.size()];
    }
    return launch_stream;
  };

  // Apply pre-computed patches -- writes HW events directly into flatPacketData
  // via the flat_packet pointers resolved at instantiate time, so no rebuild needed.
  if (!sync_plan_.patch_list.empty()) {
    device->ApplyHwEventPatches(sync_plan_.patch_list, segment_hw_events);
  }

  // Single AccumulateCommand on launch_stream manages all HW event lifetimes
  // and serves as the dispatch anchor for all segments across all streams.
  // Kernel dispatch records are added to the command at dispatch time (via
  // addKernelDispatch in dispatchAqlPacketBatchFlat) — no string borrowing,
  // no GraphExecBase pin.
  auto* graph_accumulate = new amd::AccumulateCommand(*launch_stream, {}, nullptr);

  // Register HW events with graph_accumulate so profiling can read them.
  for (auto& hw_event : segment_hw_events) {
    if (hw_event != nullptr) {
      graph_accumulate->addHwEvent(hw_event, device);
    }
  }

  // For the recycling (top-level) path, the pool owns the signals: hand the set
  // back to the caller, which forwards it to the completion callback
  // (OnLaunchComplete) that re-arms and returns it to the pool. Tell the
  // AccumulateCommand destructor not to destroy them. The legacy path keeps the
  // default (destructor destroys the locally created signals).
  if (recycle && !segment_hw_events.empty()) {
    graph_accumulate->setOwnsHwEvents(false);
    *out_signal_set = segment_hw_events;
  }

  // Process segments level by level
  for (int level = 0; level <= max_dependency_level_; ++level) {
    auto level_it = segments_per_level_.find(level);
    if (level_it == segments_per_level_.end()) {
      continue;
    }

    const auto& segments_at_level = level_it->second;

    if (level == 0) {
      // Synchronize internal streams with launch stream's last command if available
      amd::Command* launch_last_cmd = launch_stream->getLastQueuedCommand(true);
      if (launch_last_cmd != nullptr) {
        amd::Command::EventWaitList launch_wait_list;
        launch_wait_list.push_back(launch_last_cmd);

        // For each segment at level 0, if it's on a different stream, add a wait marker
        for (int segment_id : segments_at_level) {
          hip::Stream* seg_stream = resolveSegmentStream(segments_[segment_id]);
          if (seg_stream != launch_stream) {
            auto marker = new amd::Marker(*seg_stream, true, launch_wait_list);
            if (marker != nullptr) {
              marker->enqueue();
              marker->release();
            }
          }
        }
        launch_last_cmd->release();
      }
    }

    // Dispatch each segment -- barriers are in the batch, signals are patched
    for (int segment_id : segments_at_level) {
      const auto& segment = segments_[segment_id];
      hip::Stream* current_stream = resolveSegmentStream(segment);

      status = EnqueueSegment(segment, current_stream, graph_accumulate);

      if (status != hipSuccess) {
        graph_accumulate->release();
        if (out_status != nullptr) {
          *out_status = status;
        }
        return nullptr;
      }
    }
  }

  // Sync parallel-stream leaves back to launch_stream via graph_accumulate's
  // dep_signal[]. Same-stream leaves rely on in-order queue semantics instead.
  if (IsLeafNodeSyncRequired()) {
    for (int seg_id : sync_plan_.leaf_segment_ids) {
      if (seg_id < 0 || seg_id >= static_cast<int>(segments_.size())) continue;
      hip::Stream* seg_stream = resolveSegmentStream(segments_[seg_id]);
      if (seg_stream == launch_stream) continue;
      // A leaf's TAIL always has a slot (ComputeProducerNodes adds it under
      // IsLeafNodeSyncRequired); guard defensively.
      int hw_slot = HwEventSlotFor(segments_[seg_id].last_node);
      if (hw_slot < 0 || hw_slot >= static_cast<int>(segment_hw_events.size())) continue;
      graph_accumulate->addDepHwEvent(segment_hw_events[hw_slot]);
    }
  }

  graph_accumulate->enqueue();

  if (out_status != nullptr) {
    *out_status = status;
  }
  return graph_accumulate;
}

// ================================================================================================
// Graph segment to queue dispatch matching
hipError_t GraphExecSegmented::EnqueueSegment(const Segment& segment, hip::Stream* stream,
                                     amd::AccumulateCommand* accumulate) {
  hipError_t status = hipSuccess;

  // Find the SegmentBatch for this segment using O(1) map lookup
  SegmentBatch* segBatch = nullptr;
  auto segBatchIt = segmentBatches_.find(segment.id);
  if (segBatchIt != segmentBatches_.end()) {
    segBatch = &segBatchIt->second;
  }

  size_t batchIndex = 0;

  // Lambda to dispatch the current batch at batchIndex.
  // attach_signal=true asks the dispatcher to give the last packet a real
  // completion signal (via Barriers().ActiveSignal) so a downstream
  // uncaptured node — typically an SDMA memcpy on a different engine — can
  // wait on it directly through HwQueueTracker::WaitingSignal.
  auto dispatchCurrentBatch = [&](bool attach_signal = false) -> hipError_t {
    if (!segBatch || batchIndex >= segBatch->packet_batches.size()) {
      return hipSuccess;
    }
    auto& packetBatch = segBatch->packet_batches[batchIndex];
    if (packetBatch.dispatchPackets.empty()) {
      ++batchIndex;
      return hipSuccess;
    }

    const amd::AlignedVector64<uint8_t>* flatData;
    const std::vector<uint32_t>* flatHdrs;
    const std::vector<uint8_t>* metaData = nullptr;

    if (packetBatch.disabledNodeCount == 0) {
      flatData = &packetBatch.flatPacketData;
      flatHdrs = &packetBatch.validPacketFullHeaders;
      if (!packetBatch.flatMetadataData.empty()) {
        metaData = &packetBatch.flatMetadataData;
      }
    } else {
      // Guard against stale filtered buffers: rebuildFlatBuffer (called from
      // UpdateAQLPacket) invalidates the cache. This is a no-op when valid.
      packetBatch.rebuildFilteredLists(sync_plan_.patch_list);
      flatData = &packetBatch.filteredFlatPacketData;
      flatHdrs = &packetBatch.filteredValidPacketFullHeaders;
      if (!packetBatch.filteredFlatMetadataData.empty()) {
        metaData = &packetBatch.filteredFlatMetadataData;
      }
    }

    if (!flatData->empty()) {
      bool batchStatus = stream->vdev()->dispatchAqlPacketBatchFlat(
          *flatData, *flatHdrs, accumulate, attach_signal, true, false, metaData);
      if (!batchStatus) {
        return hipErrorUnknown;
      }
    }

    ++batchIndex;
    return hipSuccess;
  };

  // Handle child graph segments - recursively enqueue the entire child graph
  if (segment.child_graph_ptr != nullptr) {
    // Dispatch dependency barriers before child graph execution
    status = dispatchCurrentBatch();
    if (status != hipSuccess) return status;

    auto childGraphExec = dynamic_cast<GraphExecSegmented*>(segment.child_graph_ptr);
    if (childGraphExec != nullptr) {
      // Child graphs share the same kernel arg manager as parent (for packet capture)
      if (childGraphExec->GetKernelArgManager() == nullptr) {
        auto kernArgMgr = GetKernelArgManager();
        if (kernArgMgr != nullptr) {
          kernArgMgr->retain();
          childGraphExec->SetKernelArgManager(kernArgMgr);
        }
      }

      // Recursively enqueue the child graph with its own dependency tracking.
      // TODO: child graphs currently take the legacy create/destroy signal path
      // (out_signal_set == nullptr -> recycle == false), so their pre-created
      // signal pool (from the child's BuildSyncPlan/Prepopulate) sits unused and
      // they pay signal_create/destroy every launch. To pool child signals too,
      // pass an out_signal_set here and recycle it from the parent's
      // OnLaunchComplete (the parent's accumulate completion encloses the
      // child's work); the cleanup would carry per-pool (manager, set) pairs.
      hipError_t child_status = hipSuccess;
      amd::Command* child_last_cmd =
          childGraphExec->EnqueueSegmentedGraph(stream, {}, &child_status);

      if (child_status != hipSuccess) {
        ClPrint(amd::LOG_ERROR, amd::LOG_CODE,
                "[hipGraph] EnqueueSegment: Failed to enqueue child graph, status=%d",
                child_status);
        return child_status;
      }

      if (child_last_cmd != nullptr) {
        child_last_cmd->release();
      }
    }

    // Dispatch completion barrier after child graph — signals parent's HW event
    while (segBatch && batchIndex < segBatch->packet_batches.size()) {
      status = dispatchCurrentBatch();
      if (status != hipSuccess) return status;
    }

    return hipSuccess;
  }

  // Dispatch the leading batch (cross-dep barrier from BuildSyncPlan) before
  // the uncaptured first node so the AQL barrier precedes its GPU commands.
  if (segBatch && !segBatch->node_capture_status.empty() &&
      !segBatch->node_capture_status[0] &&
      batchIndex < segBatch->packet_batches.size()) {
    status = dispatchCurrentBatch();
    if (status != hipSuccess) return status;
  }

  // Process all nodes in this segment
  for (size_t i = 0; i < segment.nodes.size(); ++i) {
    auto& node = segment.nodes[i];

    if (segBatch && i < segBatch->node_capture_status.size() &&
        segBatch->node_capture_status[i]) {
      // Node was successfully captured - dispatch its batch
      if (segBatch && batchIndex < segBatch->packet_batches.size()) {
        auto& packetBatch = segBatch->packet_batches[batchIndex];
        if (DEBUG_HIP_GRAPH_DOT_PRINT) {
          for (size_t j = i; j < i + packetBatch.nodeRanges.size(); j++) {
            segment.nodes[j]->stream_id_ = stream->GetStreamId();
            segment.nodes[j]->hw_queue_id_ = stream->getQueueID();
          }
        }
        // Skip all consecutive captured nodes that belong to this batch
        i += packetBatch.nodeRanges.size() - 1;
        // Check if the next uncaptured node is an SDMA memcpy that needs handoff.
        // Only set sdma_follows if that's the case and the node will use SDMA.
        // Otherwise, staging-blit or other paths do not need the signal.
        bool sdma_follows = false;
        size_t next = i + 1;
        if (next < segment.nodes.size() &&
            next < segBatch->node_capture_status.size() &&
            !segBatch->node_capture_status[next] &&
            segment.nodes[next]->GetType() == hipGraphNodeTypeMemcpy) {
          auto* memcpyNode = dynamic_cast<GraphMemcpyNode*>(segment.nodes[next]);
          // Memcpy node types that don't derive from GraphMemcpyNode
          // (e.g. GraphDrvMemcpyNode) fall back to the conservative
          // "assume SDMA" behavior.
          sdma_follows = (memcpyNode == nullptr) || !memcpyNode->WillBypassSdmaEngine();
        }
        if (sdma_follows && !packetBatch.dispatchPackets.empty()) {
          stream->vdev()->addSystemScope();
        }
        status = dispatchCurrentBatch(sdma_follows);
        if (status != hipSuccess) return status;
      }
    } else {
      // Node doesn't support capture - execute individually. Upstream flat batches
      // hand off via attach_signal (SDMA memcpy) or BuildSyncPlan dep barriers;
      // no pre/post Markers needed around the uncaptured node.
      if (DEBUG_HIP_GRAPH_DOT_PRINT) {
        node->stream_id_ = stream->GetStreamId();
        node->hw_queue_id_ = stream->getQueueID();
      }
      node->SetStream(stream);
      status = node->CreateCommand(node->GetQueue());
      if (status != hipSuccess) return status;
      status = node->EnqueueCommands(stream);
      if (status != hipSuccess) return status;
    }
  }

  // Dispatch any remaining batches (e.g. trailing completion barrier for segments
  // with uncaptured nodes where the last node was non-captured)
  if (segBatch) {
    while (batchIndex < segBatch->packet_batches.size()) {
      status = dispatchCurrentBatch();
      if (status != hipSuccess) return status;
    }
  }

  return status;
}

// ================================================================================================
void GraphExecBase::UpdateStreams(hip::Stream* launch_stream) {
  int devId = (launch_stream != nullptr) ? launch_stream->vdev()->device().index()
                                         : captureDeviceId_;
  streams_.clear();
  if (launch_stream != nullptr) {
    streams_.push_back(launch_stream);
  }
  if (parallel_streams_.find(devId) == parallel_streams_.end()) {
    if (launch_stream == nullptr) {
      LogPrintfError("UpdateStreams failed for capture device id:%d", devId);
    }
    return;
  }
  auto& parallel_streams = parallel_streams_[devId];

  // Collect queue IDs already in use, starting with the launch stream if present.
  std::unordered_set<uint64_t> used_qids;
  hip::Stream* skipped_stream = nullptr;
  if (launch_stream != nullptr) {
    used_qids.insert(launch_stream->getQueueID());
    // The launch stream fills slot 0 here, so a stream added for cross-device use
    // stays idle.
    skipped_stream = (devId == captureDeviceId_) ? cross_device_stream_ : nullptr;
  }

  for (auto* stream : parallel_streams) {
    if (stream == skipped_stream) {
      continue;
    }
    uint64_t qid = stream->getQueueID();
    if (used_qids.count(qid) > 0) {
      // Collision: this stream shares a HW queue with the launch stream or another
      // internal stream. Re-acquire a different queue, avoiding all used ones.
      if (stream->vdev()->ReacquireQueueExcluding(used_qids)) {
        qid = stream->getQueueID();
        ClPrint(amd::LOG_INFO, amd::LOG_CODE,
                "[hipGraph] Resolved queue collision: stream reassigned to queueID %lu", qid);
      } else {
        ClPrint(amd::LOG_WARNING, amd::LOG_CODE,
                "[hipGraph] Could not resolve queue collision for stream (best-effort)");
      }
    }
    used_qids.insert(qid);
    streams_.push_back(stream);
  }
}

// ================================================================================================
hipError_t Graph::RunOneNode(Node node) {
  // Clear the storage of the wait nodes
  memset(&wait_order_[0], 0, sizeof(Node) * wait_order_.size());
  amd::Command::EventWaitList waitList;
  auto releaseWaitOrderCommands = [&]() {
    for (auto dep : wait_order_) {
      if (dep != nullptr) {
        for (auto command : dep->GetCommands()) {
          command->release();
        }
      }
    }
  };
  // Walk through dependencies and find the last launches on each parallel stream
  for (auto depNode : node->GetDependencies()) {
    // Process only the nodes that have been submitted
    if (depNode->launch_id_ != -1) {
      // Child graph nodes may internally dispatch work on streams other
      // than their assigned stream_id_, so the same-stream in-order
      // assumption does not hold.  Always treat them as cross-stream deps.
      if (depNode->stream_id_ != node->stream_id_ ||
          depNode->GetType() == hipGraphNodeTypeGraph) {
        // If there is no wait node on the stream, then assign one
        if ((wait_order_[depNode->stream_id_] == nullptr) ||
            // If another node executed on the same stream, then use the latest launch only,
            // since the same stream has in-order run
            (wait_order_[depNode->stream_id_]->launch_id_ < depNode->launch_id_)) {
          wait_order_[depNode->stream_id_] = depNode;
        }
      } else {
        // Release nodes that were enqueued on the same stream, since they are not included in the
        // wait list. Their references were retained for all outgoing edges.
        for (auto command : depNode->GetCommands()) {
          command->release();
        }
      }
    } else {
      node->SetWait(false);
      // It should be a safe return,
      // since the last edge to this dependency has to submit the command
      return hipSuccess;
    }
  }

  // Create a wait list from the last launches of all dependencies
  for (auto dep : wait_order_) {
    if (dep != nullptr) {
      for (auto command : dep->GetCommands()) {
        waitList.push_back(command);
      }
    }
  }
  if (node->GetType() == hipGraphNodeTypeGraph) {
    // Process child graph separately, since there is no connection
    auto child = reinterpret_cast<hip::ChildGraphNode*>(node)->GetChildGraph();
    if (!reinterpret_cast<hip::ChildGraphNode*>(node)->GetGraphCaptureStatus()) {
      auto status = child->RunNodes(node->stream_id_, &streams_, &waitList,
                                    cross_stream_producers_.count(node) != 0);
      if (status != hipSuccess) {
        releaseWaitOrderCommands();
        return status;
      }
      // Store the child graph's completion command so that downstream
      // dependency handling can use node->GetCommands() directly,
      // instead of querying getLastQueuedCommand at dependency time
      // (which could return unrelated later work on the same stream).
      auto completion = streams_[node->stream_id_]->getLastQueuedCommand(true);
      if (completion != nullptr) {
        // Release any previously stored completion command (from prior launches)
        for (auto cmd : node->GetCommands()) {
          cmd->release();
        }
        node->GetCommands().clear();
        node->GetCommands().push_back(completion);
      }
    }
  } else {
    // Assign a stream to the current node
    node->SetStream(streams_);
    if (DEBUG_HIP_GRAPH_DOT_PRINT) {
      node->hw_queue_id_ = node->GetQueue()->getQueueID();
    }
    // Create the execution commands on the assigned stream
    auto status = node->CreateCommand(node->GetQueue());
    if (status != hipSuccess) {
      LogPrintfError("Command creation for node id(%d) failed!", current_id_ + 1);
      releaseWaitOrderCommands();
      return status;
    }
    // A command with no completion signal is covered instead by the marker
    // Event::notifyCmdQueue() adds.
    const bool cross_stream = cross_stream_producers_.count(node) != 0;
    for (auto command : node->GetCommands()) {
      command->setCrossStreamProducer(cross_stream);
    }
    // If a wait was requested, then process the list
    if (node->GetWait() && !waitList.empty()) {
      node->UpdateEventWaitLists(waitList);
    }
    // Start the execution
    status = node->EnqueueCommands(node->GetQueue());
    if (status != hipSuccess) {
      releaseWaitOrderCommands();
      return status;
    }
  }
  // Release commands of dependency nodes that were included in the wait list after enqueue
  releaseWaitOrderCommands();
  // Assign the launch ID of the submitted node
  // This is also applied to childGraphs to prevent them from being reprocessed
  node->launch_id_ = current_id_++;
  uint32_t i = 0;
  // Execute the nodes in the edges list
  for (auto edge : node->GetEdges()) {
    // Don't wait in the nodes, executed on the same streams and if it has just one dependency
    bool wait =
        ((i < DEBUG_HIP_FORCE_GRAPH_QUEUES) || (edge->GetDependencies().size() > 1)) ? true : false;
    edge->SetWait(wait);
    i++;
    // Retain the current node for all its outgoing edges.
    // Each edge will include this node in its waitlist and release it after their commands are
    // enqueued.
    for (auto command : node->GetCommands()) {
      command->retain();
    }
  }
  if (node->GetEdges().size() == 0) {
    // Add a leaf node into the list for a wait.
    // Always use the last node, since it's the latest for the particular queue
    leafs_[node->stream_id_] = node;
    // An extra retain is needed for the leaves in order to be able to later enqueue a marker
    // on the app stream that has these commands in the waitlist.
    // Child graph nodes now have completion commands stored via GetCommands(),
    // so they participate in the leaf retain/release cycle like regular nodes.
    for (auto command : node->GetCommands()) {
      command->retain();
    }
  }

  node->SetWait(false);
  return hipSuccess;
}

// A hint, not an exact set: a missing entry costs a host resident wait, an extra one costs a
// barrier packet.
void Graph::FindCrossStreamProducers(int32_t base_stream) {
  cross_stream_producers_.clear();
  for (auto node : vertices_) {
    // The command a consumer of a child graph waits on is the last one the child queued,
    // which is on the child graph node's own stream.
    for (auto dep : node->GetDependencies()) {
      if (dep->stream_id_ != node->stream_id_) {
        cross_stream_producers_.insert(dep);
      }
    }
    // The join at the bottom of RunNodes() waits on each other stream's last leaf.  Which leaf
    // that is depends on traversal order, so mark every leaf off the base stream.
    if (node->GetEdges().empty() && node->stream_id_ != base_stream) {
      cross_stream_producers_.insert(node);
    }
  }
}

// ================================================================================================
hipError_t Graph::RunNodes(int32_t base_stream, const std::vector<hip::Stream*>* parallel_streams,
                           const amd::Command::EventWaitList* parent_waitlist,
                           bool waited_cross_stream) {
  if (parallel_streams != nullptr) {
    streams_ = *parallel_streams;
  }
  // Rebuilding this per launch would put a container clear/insert on a path that takes no
  // lock.  Safe only because a graph has one pending launch at a time.
  if (cross_stream_base_ != base_stream) {
    FindCrossStreamProducers(base_stream);
    cross_stream_base_ = base_stream;
  }

  // childgraph node has dependencies on parent graph nodes from other streams
  if (parent_waitlist != nullptr) {
    auto start_marker = new amd::Marker(*streams_[base_stream], true, *parent_waitlist);
    start_marker->enqueue();
    start_marker->release();
  }
  amd::Command::EventWaitList wait_list;
  current_id_ = 0;
  memset(&leafs_[0], 0, sizeof(Node) * leafs_.size());

  // Add possible waits in parallel streams for the app's default launch stream
  constexpr bool kRetainCommand = true;
  auto last_command = streams_[base_stream]->getLastQueuedCommand(kRetainCommand);
  if (last_command != nullptr) {
    // Add the last command into the waiting list
    wait_list.push_back(last_command);
    // Check if the graph has multiple root nodes
    for (uint32_t i = 0; i < DEBUG_HIP_FORCE_GRAPH_QUEUES; ++i) {
      if ((base_stream != i) && (roots_[i] != nullptr)) {
        // Wait for the app's queue
        auto start_marker = new amd::Marker(*streams_[i], true, wait_list);
        start_marker->enqueue();
        start_marker->release();
      }
    }
    // For child graphs launched on a non-zero base_stream, the root nodes
    // are on stream 0 (roots_[0] is never set because scheduling always
    // assigns the first root to stream 0 and skips it in root recording).
    // Sync stream 0 with base_stream so the child's work waits for the
    // parent's dependencies.
    if (base_stream != 0) {
      auto start_marker = new amd::Marker(*streams_[0], true, wait_list);
      start_marker->enqueue();
      start_marker->release();
    }
    last_command->release();
  }

  // Run all commands in the graph
  for (auto node : GetTopoOrder()) {
    node->launch_id_ = -1;
    auto status = RunOneNode(node);
    if (status != hipSuccess) {
      return status;
    }
  }
  wait_list.clear();
  // Check if the graph has multiple leaf nodes
  for (uint32_t i = 0; i < DEBUG_HIP_FORCE_GRAPH_QUEUES; ++i) {
    if (leafs_[i] != nullptr) {
      for (auto command : leafs_[i]->GetCommands()) {
        if (base_stream != i) {
          wait_list.push_back(command);
        } else {
          command->release();
        }
      }
    }
  }
  // Wait for leafs in the graph's app stream
  if (wait_list.size() > 0) {
    auto end_marker = new amd::Marker(*streams_[base_stream], true, wait_list);
    // RunOneNode() collects this marker only after the child has been enqueued, so the parent
    // cannot mark it; the flag comes down from there instead.
    end_marker->setCrossStreamProducer(waited_cross_stream);
    end_marker->enqueue();
    end_marker->release();
    for (auto command : wait_list) {
      command->release();
    }
  }

  return hipSuccess;
}

// ================================================================================================
hipError_t GraphExecSegmented::Run(hip::Stream* launch_stream) {
  hipError_t status = hipSuccess;

  // Retain under shared lock so hipDeviceGraphMemTrim's refcount check is accurate.
  // The lock blocks only while trim holds the exclusive (write) lock.
  {
    std::shared_lock<std::shared_mutex> trim_guard(graphExecTrimLock_);
    this->retain();
  }

  // Get the first node
  Node firstNode = nullptr;
  if (!segments_.empty() && !segments_[0].nodes.empty()) {
    firstNode = segments_[0].nodes[0];
  } else if (!topoOrder_.empty()) {
    firstNode = topoOrder_[0];
  }

  if (flags_ & hipGraphInstantiateFlagAutoFreeOnLaunch) {
    if (firstNode != nullptr) {
      auto* parentGraph = firstNode->GetParentGraph();
      auto* pool = parentGraph->Device()->GetGraphMemoryPool();
      for (auto* node : topoOrder_) {
        if (node->GetType() == hipGraphNodeTypeMemAlloc) {
          static_cast<GraphMemAllocNode*>(node)->ReleaseCachedMapping(pool, launch_stream);
        }
      }
      parentGraph->FreeAllMemory(launch_stream);
      parentGraph->memalloc_nodes_ = 0;
      if (!AMD_DIRECT_DISPATCH) {
        // The MemoryPool::FreeAllMemory queues a memory unmap command that for !AMD_DIRECT_DISPATCH
        // runs asynchonously. Make sure that freeAllMemory is complete before creating new commands
        // to prevent races to the MemObjMap.
        launch_stream->finish();
      }
    }
  }

  // If this is a repeat launch, make sure corresponding MemFreeNode exists for a MemAlloc node
  if (repeatLaunch_ == true) {
    if (firstNode != nullptr && firstNode->GetParentGraph()->GetMemAllocNodeCount() > 0) {
      this->release();
      return hipErrorInvalidValue;
    }
  } else {
    repeatLaunch_ = true;
  }

  ClPrint(amd::LOG_DEBUG, amd::LOG_CODE, "GraphExecSegmented::Run max_streams: %d, on device: %d",
          max_streams_, launch_stream->DeviceId());

  // If the launch stream lost its HW queue due to dynamic queue management,
  // try to re-acquire the same one it used last time.
  // Then run collision detection to ensure graph-internal streams don't share
  // a HW queue with the launch stream.
  const bool cross_device_launch = (captureDeviceId_ != launch_stream->DeviceId());
  launch_stream->vdev()->SetPreferredQueue();
  launch_stream->vdev()->AcquireQueueWithPreference();
  // Slot 0 must be an internal capture-device stream when the launch stream lives on
  // another device. Built on first use; later launches only test the pointer.
  if (cross_device_launch && cross_device_stream_ == nullptr) {
    status = EnsureCrossDeviceStream();
    if (status != hipSuccess) {
      this->release();
      return status;
    }
  }
  UpdateStreams(cross_device_launch ? nullptr : launch_stream);

  // Signals borrowed from the per-graph pool for this launch (segmented path
  // only); handed to the completion callback to re-arm and return to the pool.
  std::vector<void*> launch_signal_set;

  // Command whose completion drives OnLaunchComplete. On the segmented path we
  // reuse the graph's own accumulate command instead of enqueuing a dedicated marker
  amd::Command* completion_cmd = nullptr;

  // If the graph has kernels that do device-side allocation, packet capture
  // needs the hidden heap on the graph/capture device, not necessarily the
  // user-visible launch stream device.
  hip::Stream* graph_launch_stream = cross_device_launch ? streams_[0] : launch_stream;
  if (HasHiddenHeap() &&
      hiddenHeapInitializedDevices_.insert(captureDeviceId_).second) {
    graph_launch_stream->vdev()->HiddenHeapInit();
  }

  amd::Command* last_cmd = nullptr;
  if (!cross_device_launch) {
    if (max_streams_dev_.size() == 1) {
      // Single-device: pass collision-handled streams_ to EnqueueSegmentedGraph
      last_cmd = EnqueueSegmentedGraph(launch_stream, streams_, &status, &launch_signal_set);
    } else {
      // Multi-device: pass empty vector, will use parallel_streams_ internally
      last_cmd = EnqueueSegmentedGraph(launch_stream, {}, &status, &launch_signal_set);
    }
  } else {
    // Cross-device launch: replay segmented AQL on a capture-device stream.
    // A marker on the graph stream preserves prior foreign launch-stream
    // ordering when such work exists. The completion marker below makes graph
    // completion visible on the foreign launch stream.
    hip::Stream* graph_stream = streams_[0];
    constexpr bool kRetainCommand = true;
    amd::Command* launch_last_cmd = launch_stream->getLastQueuedCommand(kRetainCommand);
    if (launch_last_cmd != nullptr) {
      amd::Command::EventWaitList launch_wait_list;
      launch_wait_list.push_back(launch_last_cmd);
      auto* graph_start = new amd::Marker(*graph_stream, true, launch_wait_list);
      graph_start->enqueue();
      graph_start->release();
      launch_last_cmd->release();
    }

    last_cmd = EnqueueSegmentedGraph(graph_stream, streams_, &status, &launch_signal_set);
    if (status == hipSuccess && last_cmd != nullptr) {
      amd::Command::EventWaitList completion_wait_list;
      completion_wait_list.push_back(last_cmd);

      auto* launch_done = new amd::Marker(*launch_stream, kMarkerDisableFlush, completion_wait_list);
      launch_done->enqueue();
      // The launch stream queue owns this marker now. Keep last_cmd as the
      // capture-device completion command that drives graph resource cleanup.
      launch_done->release();
    } else if (status != hipSuccess) {
      // Error path only: EnqueueSegmentedGraph may have queued partial work
      // before failing. Drain the capture-device streams before the common
      // cleanup path can recycle launch signals or drop this exec reference.
      for (auto* stream : streams_) {
        if (stream != nullptr) {
          stream->finish();
        }
      }
    }
  }

  // Drive OnLaunchComplete off this command's completion (its leaf-sync deps
  // already imply all parallel work is done). Our reference is released after
  // the callback is registered below; the queue keeps it alive until done.
  completion_cmd = last_cmd;
  if (DEBUG_HIP_GRAPH_DOT_PRINT == 2 && !graph_dumped_) {
    graph_dumped_ = true;
    std::string filename =
        "graph_" + std::to_string(amd::Os::getProcessId()) + "_dot_print_launch_1";
    hipError_t status = ihipGraphDebugDotPrint(this, filename.c_str(), 0);
    if (status == hipSuccess) {
      ClPrint(amd::LOG_DETAIL_DEBUG, amd::LOG_CODE, "[hipGraph] graph dump:%s", filename.c_str());
    }
  }
  // Register the refcount/recycle callback. Prefer the graph's own accumulate
  // command (segmented path); otherwise enqueue a lightweight marker just to
  // carry the callback.
  amd::Command* CallbackCommand = completion_cmd;
  const bool own_callback_cmd = (CallbackCommand == nullptr);
  if (own_callback_cmd) {
    CallbackCommand = new amd::Marker(*graph_launch_stream, kMarkerDisableFlush, {});
    // we may not need to flush any caches.
    CallbackCommand->setCommandEntryScope(amd::Device::kCacheStateIgnore);
  }
  amd::Event& event = CallbackCommand->event();
  constexpr bool kBlocking = false;
  auto* cleanup = new GraphLaunchCleanup();
  cleanup->exec = this;
  cleanup->device = g_devices[captureDeviceId_]->devices()[0];
  cleanup->signal_set = std::move(launch_signal_set);
  if (!event.setCallback(CL_COMPLETE, GraphExecBase::OnLaunchComplete, cleanup, kBlocking)) {
    // setCallback essentially never fails, but if it does the launch's GPU work
    // is already queued (the accumulate is enqueued and was told not to destroy
    // its signals). Drain that work, then run the same completion handling the
    // callback would have (recycle the borrowed pooled signals + drop our
    // reference) so they are not leaked and the pool does not shrink.
    graph_launch_stream->finish();
    OnLaunchComplete(nullptr, CL_COMPLETE, cleanup);
    CallbackCommand->release();
    return hipErrorInvalidHandle;
  }
  // The marker must be enqueued to run; the accumulate command is already
  // enqueued by EnqueueSegmentedGraph. Either way, release our reference: the
  // queue keeps the command alive until completion, when the callback fires.
  if (own_callback_cmd) {
    CallbackCommand->enqueue();
  }
  CallbackCommand->release();
  return status;
}

// ================================================================================================
// Process-wide count of signal sets created OUTSIDE prepopulation, i.e. pool growth.
static std::atomic<size_t> g_graph_signal_sets_created{0};

GraphSignalManager::~GraphSignalManager() {
  // No launches can be in flight at this point (GraphExecBase refcount guarantees
  // it outlives all launches), so every set is back in the free pool.
  for (auto& dev_pool : free_sets_) {
    amd::Device* device = dev_pool.first;
    for (auto& set : dev_pool.second) {
      // Pooled signals rest armed (value 1); mark them idle before destroy so
      // ~ProfilingSignal does not block waiting on an armed-but-idle signal.
      device->QuiesceHwEvents(set);
      for (void* sig : set) {
        if (sig != nullptr) {
          // Pair with CreateHwEvents() so non-ROCm devices can hook teardown.
          device->DestroyHwEvent(sig);
        }
      }
    }
  }
  free_sets_.clear();
}

bool GraphSignalManager::Prepopulate(amd::Device* device, int count, int num_sets) {
  if (count <= 0 || num_sets <= 0) {
    return true;
  }
  std::lock_guard<std::mutex> lock(lock_);
  auto& pool = free_sets_[device];

  // BuildSyncPlan is re-runnable, so Prepopulate may be called more than once.
  // If a prior run sized the sets for a different segment count, those sets are
  // unusable -- destroy and rebuild. No launches are in flight at (re)instantiate
  // time, so every set for this device is present in the free pool here.
  if (!pool.empty() && static_cast<int>(pool.back().size()) != count) {
    for (auto& set : pool) {
      device->QuiesceHwEvents(set);
      for (void* sig : set) {
        if (sig != nullptr) {
          device->DestroyHwEvent(sig);
        }
      }
    }
    pool.clear();
  }

  // Top up to num_sets only; do not unconditionally append on every call, which
  // would grow the pool without bound across re-instantiations.
  const int before = static_cast<int>(pool.size());
  for (int i = before; i < num_sets; ++i) {
    std::vector<void*> set;
    if (!device->CreateHwEvents(count, set)) {
      return false;
    }
    pool.push_back(std::move(set));
  }
  ClPrint(amd::LOG_INFO, amd::LOG_CODE,
          "[hipGraph] SignalPool: prepopulated %d -> %zu sets of %d signals "
          "(ordering_edge=%u, device-resident when non-zero)",
          before, pool.size(), count, static_cast<uint32_t>(DEBUG_CLR_DEVICE_ORDERING_EDGE));
  return true;
}

bool GraphSignalManager::AcquireSet(amd::Device* device, int count,
                                    std::vector<void*>& out_set) {
  if (count <= 0) {
    out_set.clear();
    return true;
  }

  {
    std::lock_guard<std::mutex> lock(lock_);
    auto& pool = free_sets_[device];
    if (!pool.empty()) {
      // Hot path: just hand out a ready (already-armed) set, then patch it.
      out_set = std::move(pool.back());
      pool.pop_back();
      return true;
    }
  }

  // Fallback only: more launches in flight than pre-created sets. Create one
  // (armed to 1 by CreateHwEvents); it joins the pool when released.
  //
  // ⚠️ This path is not free and it is not bounded. At DEBUG_CLR_DEVICE_ORDERING_EDGE != 0
  // each signal carries a DEVICE-RESIDENT value word, so every set created here consumes
  // DEVICE memory that is never returned until the graph is destroyed -- it reads as
  // "non-torch memory" to a framework doing a memory profile, not as a leak. Counted and
  // reported so pool growth is visible instead of being attributed to the model.
  const size_t grown = ++g_graph_signal_sets_created;
  if ((grown & (grown - 1)) == 0) {  // powers of two only: O(log n) lines, not O(n)
    ClPrint(amd::LOG_INFO, amd::LOG_CODE,
            "[hipGraph] SignalPoolGrowth: %zu extra set(s) created beyond prepopulation "
            "(%d signals each, ordering_edge=%u)",
            grown, count, static_cast<uint32_t>(DEBUG_CLR_DEVICE_ORDERING_EDGE));
  }
  return device->CreateHwEvents(count, out_set);
}

void GraphSignalManager::ReleaseSet(amd::Device* device, std::vector<void*>& set) {
  if (set.empty()) {
    return;
  }
  // Re-arm the signals for the next launch. Safe here because this runs from
  // the launch completion callback, so the GPU work that used them is done.
  device->ResetHwEvents(set);
  std::lock_guard<std::mutex> lock(lock_);
  free_sets_[device].push_back(std::move(set));
}

// ================================================================================================
bool GraphKernelArgManager::AllocGraphKernargPool(size_t pool_size, amd::Device* device) {
  bool bStatus = true;
  assert(pool_size > 0);
  address graph_kernarg_base;
  if (device->info().largeBar_) {
    amd::Device::AllocationFlags flags = {};
    flags.executable_ = true;
    graph_kernarg_base = reinterpret_cast<address>(device->deviceLocalAlloc(pool_size, flags));
    device_kernarg_pool_ = true;
  } else {
    graph_kernarg_base = reinterpret_cast<address>(
        device->hostAlloc(pool_size, 0, amd::Device::MemorySegment::kKernArg));
  }

  if (graph_kernarg_base == nullptr) {
    return false;
  }
  kernarg_graph_[device].push_back(KernelArgPoolGraph(graph_kernarg_base, pool_size));
  return true;
}

address GraphKernelArgManager::AllocKernArg(size_t size, size_t alignment, int devId) {
  if (size == 0) {
    return nullptr;
  }

  amd::Device* device = g_devices[devId]->devices()[0];
  assert(alignment != 0 && "Alignment must be non-zero");

  // Check if we have any pools allocated for this device
  auto& device_pools = kernarg_graph_[device];
  if (device_pools.empty()) {
    return nullptr;
  }

  auto& current_pool = device_pools.back();
  // Calculate aligned address for the allocation
  address aligned_addr = amd::alignUp(current_pool.kernarg_pool_addr_ + current_pool.kernarg_pool_offset_, alignment);
  const size_t new_pool_usage = (aligned_addr + size) - current_pool.kernarg_pool_addr_;

  // Check if allocation fits in current pool
  if (new_pool_usage <= current_pool.kernarg_pool_size_) {
    current_pool.kernarg_pool_offset_ = new_pool_usage;
    return aligned_addr;
  }

  // Current pool is full - allocate a new pool with the same size
  if (!AllocGraphKernargPool(current_pool.kernarg_pool_size_, device)) {
    return nullptr;
  }

  // Recursively allocate from the new pool
  return AllocKernArg(size, alignment, devId);
}

void GraphKernelArgManager::ReadBackOrFlush() {
  if (!device_kernarg_pool_) {
    return;
  }

  for (const auto& kernarg : kernarg_graph_) {
    const auto kernArgImpl = kernarg.first->settings().kernel_arg_impl_;

    if (kernArgImpl == KernelArgImpl::DeviceKernelArgsReadback) {
      const auto& pool = kernarg.second.back();
      if (pool.kernarg_pool_addr_ == 0) {
        continue;
      }

      // Perform readback operation on the last byte of the pool
      address dev_ptr = pool.kernarg_pool_addr_ + pool.kernarg_pool_size_;
      volatile unsigned char* sentinel_ptr = reinterpret_cast<volatile unsigned char*>(dev_ptr - 1);

      // Read-modify-write sequence with memory barriers
      volatile unsigned char kSentinel = *sentinel_ptr;
#if defined(ATI_ARCH_X86)
      _mm_sfence();
#else
      __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
      *sentinel_ptr = kSentinel;
#if defined(ATI_ARCH_X86)
      _mm_mfence();
#else
      __atomic_thread_fence(__ATOMIC_SEQ_CST);
#endif
      kSentinel = *sentinel_ptr;
      (void)kSentinel; // Suppress unused variable warning
    }
  }
}
}  // namespace hip
