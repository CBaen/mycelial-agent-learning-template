# Big Rock 7: GNN Communication Layer - Implementation Plan

**Project:** Mycelial Agent Engine (MAE) v3.0
**Phase:** Phase 1 - Weeks 3-4 (Days 15-28)
**Author:** MAE Development Team
**Date:** 2025-11-12
**Status:** Planning Phase

---

## Executive Summary

Big Rock 7 implements Graph Neural Network (GNN)-based communication for the MAE framework, providing intelligent message routing and 40-60% reduction in communication overhead. This layer learns optimal communication patterns dynamically, adapting to agent relationships and task requirements.

**Key Innovation:** Unlike traditional broadcast or point-to-point messaging, GNN communication uses learned graph representations to route messages efficiently through the most relevant paths in the agent network.

**Performance Target:** 40-60% reduction in communication overhead compared to naive broadcast approaches, while maintaining or improving coordination effectiveness.

---

## Research Foundation

### 1. Graph Neural Networks for Multi-Agent Systems

**Core Concept:**
- Agents form a dynamic graph where edges represent communication relationships
- GNN layers propagate messages through this graph, learning which connections are most valuable
- Node embeddings capture agent state, capabilities, and context
- Edge weights represent communication value/priority

**Key Papers:**
1. "Graph Neural Networks for Multi-Agent Systems" (Jiang et al., 2022)
2. "CommNet: Learning Multiagent Communication with Backpropagation" (Sukhbaatar et al., 2016)
3. "Learning to Communicate with Deep Multi-Agent Reinforcement Learning" (Foerster et al., 2016)
4. "Graph Attention Networks" (Veličković et al., 2018)

**Mycelial Inspiration:**
- Mycelial networks optimize nutrient transport through selective strengthening of high-value connections
- Our GNN mimics this by learning which agent connections are most valuable for coordination
- Like fungi, we adapt the communication graph dynamically based on resource needs and opportunities

### 2. Message Passing Neural Networks (MPNN)

**MPNN Framework:**
```
For each timestep t:
  1. Message Generation: m_ij = φ(h_i, h_j, e_ij)
  2. Message Aggregation: m_i = ⊕_{j∈N(i)} m_ij
  3. Node Update: h_i' = ψ(h_i, m_i)
```

Where:
- `h_i` = node (agent) embedding
- `e_ij` = edge (relationship) embedding
- `φ` = message function
- `⊕` = aggregation (sum, mean, max)
- `ψ` = update function

**Benefits:**
- Reduces O(n²) broadcast to O(k·n) where k = avg neighbors
- Learns which messages are relevant
- Adapts to changing agent relationships

### 3. Graph Attention Mechanisms

**Attention-Based Message Passing:**
```
α_ij = softmax(LeakyReLU(a^T [W·h_i || W·h_j]))
h_i' = σ(Σ_j α_ij W·h_j)
```

**Advantages:**
- Agents learn which neighbors to attend to
- Dynamic weighting based on context
- Interpretable attention weights show communication flow

---

## Architecture Design

### Layer 1: Graph Construction and Maintenance

**Purpose:** Dynamically build and update the agent communication graph

**Components:**

1. **AgentNode**
   ```python
   @dataclass
   class AgentNode:
       agent_id: str
       embedding: np.ndarray  # d-dimensional feature vector
       agent_type: str  # "builder", "specialist", "risk_manager"
       capabilities: Set[str]
       level: int
       position: Tuple[float, ...]  # Stigmergy position
       last_update: float
   ```

2. **CommunicationEdge**
   ```python
   @dataclass
   class CommunicationEdge:
       source_id: str
       target_id: str
       weight: float  # Learned importance [0, 1]
       message_count: int
       last_message_time: float
       success_rate: float  # Coordination success on this edge
       edge_type: str  # "collaboration", "hierarchy", "proximity"
   ```

3. **AgentGraph**
   ```python
   class AgentGraph:
       def __init__(self, embedding_dim: int = 64):
           self.nodes: Dict[str, AgentNode] = {}
           self.edges: Dict[Tuple[str, str], CommunicationEdge] = {}
           self.adjacency: Dict[str, Set[str]] = defaultdict(set)

       def add_agent(self, agent_id: str, agent_type: str, ...)
       def remove_agent(self, agent_id: str)
       def add_edge(self, source_id: str, target_id: str, ...)
       def remove_edge(self, source_id: str, target_id: str)
       def update_edge_weight(self, source_id: str, target_id: str, weight: float)
       def get_neighbors(self, agent_id: str, k: int = None) -> List[str]
       def get_subgraph(self, agent_ids: List[str]) -> 'AgentGraph'
   ```

**Graph Construction Strategies:**

1. **Proximity-Based:**
   - Connect agents within stigmergy radius
   - Edges weighted by spatial distance

2. **Role-Based:**
   - Connect agents in same Rule of 3 team
   - Connect specialists with builders needing their capabilities
   - Connect risk managers with all agents they supervise

3. **Performance-Based:**
   - Create edges between agents with successful collaboration history
   - Strengthen edges that lead to high rewards

4. **Dynamic Pruning:**
   - Remove edges with low weight (< threshold)
   - Remove inactive edges (no messages in T timesteps)
   - Maintain minimum connectivity (prevent islands)

### Layer 2: GNN Message Passing

**Purpose:** Propagate messages through the graph efficiently

**Components:**

1. **GNNMessage**
   ```python
   @dataclass
   class GNNMessage:
       message_id: str
       sender_id: str
       content: Dict[str, Any]
       message_type: str  # "request", "response", "broadcast", "query"
       priority: float  # [0, 1]
       ttl: int  # Hops remaining
       path: List[str]  # Agents visited
       timestamp: float
   ```

2. **MessageEncoder**
   ```python
   class MessageEncoder:
       """Encode messages into fixed-size embeddings"""
       def __init__(self, embedding_dim: int = 64):
           self.embedding_dim = embedding_dim
           self.content_encoder = self._build_encoder()

       def encode(self, message: GNNMessage, sender_node: AgentNode) -> np.ndarray:
           # Combine message content, type, priority, sender embedding
           content_features = self._encode_content(message.content)
           type_features = self._encode_type(message.message_type)
           meta_features = np.array([message.priority, message.ttl])

           return np.concatenate([
               sender_node.embedding,
               content_features,
               type_features,
               meta_features
           ])
   ```

3. **GNNLayer**
   ```python
   class GNNLayer:
       """Single GNN propagation layer"""
       def __init__(self, embedding_dim: int = 64, num_heads: int = 4):
           self.embedding_dim = embedding_dim
           self.num_heads = num_heads

           # Learnable parameters
           self.W_message = np.random.randn(embedding_dim, embedding_dim)
           self.W_update = np.random.randn(embedding_dim, embedding_dim)
           self.attention_weights = np.random.randn(num_heads, embedding_dim)

       def forward(
           self,
           node_embeddings: Dict[str, np.ndarray],
           adjacency: Dict[str, Set[str]],
           messages: Dict[str, List[GNNMessage]]
       ) -> Dict[str, np.ndarray]:
           """
           Message passing step:
           1. For each node, aggregate messages from neighbors
           2. Apply attention to weight neighbor contributions
           3. Update node embeddings
           """
           new_embeddings = {}

           for node_id, embedding in node_embeddings.items():
               # Get messages from neighbors
               neighbor_messages = self._aggregate_neighbor_messages(
                   node_id, adjacency, messages
               )

               # Compute attention weights
               attention = self._compute_attention(
                   embedding, neighbor_messages
               )

               # Weighted aggregation
               aggregated = self._weighted_aggregate(
                   neighbor_messages, attention
               )

               # Update embedding
               new_embeddings[node_id] = self._update_embedding(
                   embedding, aggregated
               )

           return new_embeddings

       def _compute_attention(
           self,
           node_embedding: np.ndarray,
           neighbor_embeddings: List[np.ndarray]
       ) -> np.ndarray:
           """Compute multi-head attention weights"""
           attention_scores = []

           for head in range(self.num_heads):
               # Dot product attention
               scores = [
                   np.dot(
                       node_embedding,
                       self.attention_weights[head]
                   ) * np.dot(
                       neighbor_emb,
                       self.attention_weights[head]
                   )
                   for neighbor_emb in neighbor_embeddings
               ]
               attention_scores.append(self._softmax(np.array(scores)))

           # Average over heads
           return np.mean(attention_scores, axis=0)
   ```

4. **GNNMessagePropagator**
   ```python
   class GNNMessagePropagator:
       """Orchestrates multi-layer message propagation"""
       def __init__(
           self,
           num_layers: int = 3,
           embedding_dim: int = 64,
           aggregation: str = "mean"  # "mean", "sum", "max"
       ):
           self.layers = [
               GNNLayer(embedding_dim) for _ in range(num_layers)
           ]
           self.aggregation = aggregation

       def propagate(
           self,
           graph: AgentGraph,
           message: GNNMessage,
           max_hops: int = 3
       ) -> List[str]:
           """
           Propagate message through graph using learned routing.

           Returns:
               List of agent IDs that should receive the message
           """
           recipients = []
           current_hop = [message.sender_id]
           visited = {message.sender_id}

           for hop in range(max_hops):
               next_hop = []

               for agent_id in current_hop:
                   # Get neighbors
                   neighbors = graph.get_neighbors(agent_id)

                   # Compute relevance scores using GNN
                   relevance_scores = self._compute_relevance(
                       agent_id, neighbors, message, graph
                   )

                   # Select top-k most relevant neighbors
                   selected = self._select_top_k(
                       neighbors, relevance_scores, k=3
                   )

                   for neighbor_id in selected:
                       if neighbor_id not in visited:
                           recipients.append(neighbor_id)
                           next_hop.append(neighbor_id)
                           visited.add(neighbor_id)

               if not next_hop:
                   break

               current_hop = next_hop

           return recipients

       def _compute_relevance(
           self,
           sender_id: str,
           candidate_ids: List[str],
           message: GNNMessage,
           graph: AgentGraph
       ) -> np.ndarray:
           """Compute relevance scores for candidate recipients"""
           sender_node = graph.nodes[sender_id]
           message_emb = MessageEncoder().encode(message, sender_node)

           scores = []
           for candidate_id in candidate_ids:
               candidate_node = graph.nodes[candidate_id]

               # Compute compatibility score
               score = np.dot(message_emb, candidate_node.embedding)

               # Weight by edge strength
               edge = graph.edges.get((sender_id, candidate_id))
               if edge:
                   score *= edge.weight

               scores.append(score)

           return np.array(scores)
   ```

### Layer 3: Communication Coordinator

**Purpose:** High-level API for agents to send/receive messages

**Components:**

1. **GNNCommunicator**
   ```python
   class GNNCommunicator:
       """Main communication interface for agents"""

       def __init__(
           self,
           embedding_dim: int = 64,
           max_message_history: int = 10000,
           enable_learning: bool = True
       ):
           self.graph = AgentGraph(embedding_dim)
           self.propagator = GNNMessagePropagator(num_layers=3, embedding_dim=embedding_dim)
           self.message_queue: Dict[str, deque] = defaultdict(deque)
           self.message_history: deque = deque(maxlen=max_message_history)
           self.lock = threading.RLock()

           # Learning components
           self.enable_learning = enable_learning
           self.reward_buffer: List[Tuple[str, float]] = []

       def register_agent(
           self,
           agent_id: str,
           agent_type: str,
           capabilities: Set[str],
           level: int,
           position: Tuple[float, ...]
       ):
           """Register agent in communication graph"""
           with self.lock:
               embedding = self._initialize_embedding(agent_type, capabilities, level)
               self.graph.add_agent(agent_id, agent_type, capabilities, level, position, embedding)

               # Create initial edges based on role and proximity
               self._create_initial_edges(agent_id)

       def send_message(
           self,
           sender_id: str,
           content: Dict[str, Any],
           message_type: str = "broadcast",
           target_ids: Optional[List[str]] = None,
           priority: float = 0.5,
           max_hops: int = 3
       ) -> str:
           """
           Send message through GNN routing.

           Args:
               sender_id: Sending agent
               content: Message payload
               message_type: "broadcast", "targeted", "query", "response"
               target_ids: Optional specific targets (for targeted messages)
               priority: Message priority [0, 1]
               max_hops: Maximum propagation distance

           Returns:
               Message ID
           """
           with self.lock:
               message_id = f"msg_{time.time_ns()}_{sender_id}"

               message = GNNMessage(
                   message_id=message_id,
                   sender_id=sender_id,
                   content=content,
                   message_type=message_type,
                   priority=priority,
                   ttl=max_hops,
                   path=[sender_id],
                   timestamp=time.time()
               )

               # Determine recipients using GNN routing
               if target_ids:
                   recipients = self._route_to_targets(message, target_ids)
               else:
                   recipients = self.propagator.propagate(
                       self.graph, message, max_hops
                   )

               # Deliver to recipients
               for recipient_id in recipients:
                   self.message_queue[recipient_id].append(message)

               # Record for learning
               self.message_history.append({
                   'message': message,
                   'recipients': recipients,
                   'timestamp': time.time()
               })

               return message_id

       def receive_messages(
           self,
           agent_id: str,
           message_type: Optional[str] = None,
           max_messages: int = 10
       ) -> List[GNNMessage]:
           """Retrieve messages for an agent"""
           with self.lock:
               if agent_id not in self.message_queue:
                   return []

               messages = []
               queue = self.message_queue[agent_id]

               while queue and len(messages) < max_messages:
                   msg = queue.popleft()
                   if message_type is None or msg.message_type == message_type:
                       messages.append(msg)

               return messages

       def update_edge_from_outcome(
           self,
           sender_id: str,
           recipient_id: str,
           reward: float
       ):
           """Update edge weight based on communication outcome"""
           if not self.enable_learning:
               return

           with self.lock:
               edge = self.graph.edges.get((sender_id, recipient_id))
               if edge:
                   # Exponential moving average update
                   alpha = 0.1
                   edge.weight = (1 - alpha) * edge.weight + alpha * reward
                   edge.success_rate = (1 - alpha) * edge.success_rate + alpha * reward

       def optimize_graph(self):
           """Periodic graph optimization based on learned patterns"""
           with self.lock:
               # Prune weak edges
               edges_to_remove = []
               for (source, target), edge in self.graph.edges.items():
                   if edge.weight < 0.1 or edge.message_count == 0:
                       edges_to_remove.append((source, target))

               for source, target in edges_to_remove:
                   self.graph.remove_edge(source, target)

               # Strengthen successful patterns
               for (source, target), edge in self.graph.edges.items():
                   if edge.success_rate > 0.8:
                       edge.weight = min(1.0, edge.weight * 1.1)

       def get_communication_statistics(self) -> Dict[str, Any]:
           """Get communication efficiency metrics"""
           total_messages = len(self.message_history)
           total_agents = len(self.graph.nodes)
           total_edges = len(self.graph.edges)

           # Compute overhead reduction
           avg_recipients = np.mean([
               len(record['recipients'])
               for record in self.message_history
           ]) if self.message_history else 0

           # Broadcast overhead = all agents - sender
           broadcast_overhead = total_agents - 1 if total_agents > 0 else 1
           overhead_reduction = (
               1 - (avg_recipients / broadcast_overhead)
           ) * 100 if broadcast_overhead > 0 else 0

           return {
               'total_messages': total_messages,
               'total_agents': total_agents,
               'total_edges': total_edges,
               'avg_recipients_per_message': round(avg_recipients, 2),
               'broadcast_recipients': broadcast_overhead,
               'overhead_reduction_percent': round(overhead_reduction, 2),
               'avg_degree': round(2 * total_edges / total_agents, 2) if total_agents > 0 else 0
           }
   ```

### Layer 4: Integration with Existing Systems

**Integration Points:**

1. **Electrical Signaling Layer (Big Rock 5)**
   - Critical signals bypass GNN routing for immediate broadcast
   - Lower-priority signals use GNN for efficient routing
   - Signals can trigger GNN message propagation

2. **Stigmergic Environment (Big Rock 6)**
   - Use stigmergy positions for proximity-based edges
   - Marker sensing informs edge creation (agents near same markers)
   - GNN messages can trigger marker deposition

3. **Motivation & Safeguards (Big Rock 4)**
   - Successful collaborations strengthen GNN edges
   - Curiosity drives exploration messages through GNN
   - Convergence reduces communication needs

**Unified Agent API:**
```python
# In base_agent.py

def __init__(
    self,
    signal_bus: Optional[ElectricalSignalBus] = None,
    stigmergy_env: Optional[StigmergicEnvironment] = None,
    gnn_communicator: Optional[GNNCommunicator] = None
):
    self.signal_bus = signal_bus
    self.stigmergy_env = stigmergy_env
    self.gnn_communicator = gnn_communicator

    # Register with GNN if available
    if self.gnn_communicator:
        self.gnn_communicator.register_agent(
            self.unique_id,
            self.agent_type,
            self.capabilities,
            self.level,
            self.stigmergy_position
        )

def send_gnn_message(
    self,
    content: Dict[str, Any],
    message_type: str = "broadcast",
    targets: Optional[List[str]] = None,
    priority: float = 0.5
) -> Optional[str]:
    """Send message through GNN routing"""
    if not self.gnn_communicator:
        return None

    return self.gnn_communicator.send_message(
        self.unique_id,
        content,
        message_type,
        targets,
        priority
    )

def receive_gnn_messages(
    self,
    message_type: Optional[str] = None,
    max_messages: int = 10
) -> List[GNNMessage]:
    """Receive GNN-routed messages"""
    if not self.gnn_communicator:
        return []

    return self.gnn_communicator.receive_messages(
        self.unique_id,
        message_type,
        max_messages
    )

def process_gnn_messages(self):
    """Process all pending GNN messages"""
    messages = self.receive_gnn_messages()

    for message in messages:
        # Handle different message types
        if message.message_type == "collaboration_request":
            self._handle_collaboration_request(message)
        elif message.message_type == "query":
            self._handle_query(message)
        elif message.message_type == "knowledge_share":
            self._handle_knowledge_share(message)

def report_communication_outcome(
    self,
    recipient_id: str,
    success: bool,
    reward: float = 0.0
):
    """Report outcome to update GNN edges"""
    if not self.gnn_communicator:
        return

    reward_value = reward if success else -0.5
    self.gnn_communicator.update_edge_from_outcome(
        self.unique_id,
        recipient_id,
        reward_value
    )
```

---

## Performance Targets

### 1. Overhead Reduction

**Target:** 40-60% reduction in communication overhead

**Measurement:**
```python
overhead_reduction = 1 - (avg_gnn_recipients / (total_agents - 1))
```

**Scenarios:**
- 10 agents: Broadcast = 9 recipients, GNN target = 3-5 recipients (44-66% reduction)
- 50 agents: Broadcast = 49 recipients, GNN target = 15-25 recipients (49-69% reduction)
- 100 agents: Broadcast = 99 recipients, GNN target = 30-50 recipients (49-70% reduction)

### 2. Latency

**Target:** < 5ms message routing decision

**Breakdown:**
- Graph lookup: < 0.5ms
- GNN inference: < 3ms
- Recipient selection: < 1ms
- Queue insertion: < 0.5ms

### 3. Scalability

**Target:** Support 1000+ agents with < 100ms routing

**Requirements:**
- Sparse graphs (avg degree 5-10)
- Efficient neighbor queries
- Batched GNN inference
- Parallel message delivery

### 4. Learning Efficiency

**Target:** Converge to near-optimal routing within 1000 messages

**Metrics:**
- Edge weight stability
- Routing consistency
- Reward accumulation

---

## Implementation Timeline

### Week 3 (Days 15-21)

**Day 15: Core Data Structures**
- [ ] Implement `AgentNode` and `CommunicationEdge`
- [ ] Implement `AgentGraph` with CRUD operations
- [ ] Add graph construction strategies (proximity, role, performance)
- [ ] Tests: Graph operations (15 tests)

**Day 16: Message Encoding**
- [ ] Implement `GNNMessage` dataclass
- [ ] Implement `MessageEncoder` for embedding generation
- [ ] Add message type encoding
- [ ] Tests: Encoding consistency (8 tests)

**Day 17: GNN Layer**
- [ ] Implement `GNNLayer` with attention mechanism
- [ ] Add message aggregation functions (mean, sum, max)
- [ ] Implement embedding update logic
- [ ] Tests: Forward pass, attention computation (12 tests)

**Day 18: Message Propagation**
- [ ] Implement `GNNMessagePropagator`
- [ ] Add multi-hop routing logic
- [ ] Implement relevance scoring
- [ ] Tests: Routing correctness, path finding (10 tests)

**Day 19: Communication Coordinator**
- [ ] Implement `GNNCommunicator`
- [ ] Add agent registration
- [ ] Implement send/receive message APIs
- [ ] Tests: End-to-end communication (15 tests)

**Day 20: Learning & Optimization**
- [ ] Add edge weight updates from outcomes
- [ ] Implement graph optimization
- [ ] Add performance metrics
- [ ] Tests: Learning convergence, optimization (10 tests)

**Day 21: Integration & Review**
- [ ] Review all components
- [ ] Integration tests with existing systems
- [ ] Performance benchmarks
- [ ] Code cleanup

### Week 4 (Days 22-28)

**Day 22: base_agent.py Integration**
- [ ] Add GNN methods to `BaseAgent`
- [ ] Integrate with electrical signaling
- [ ] Integrate with stigmergy
- [ ] Tests: Agent-level GNN usage (12 tests)

**Day 23: Builder Agent Integration**
- [ ] Add GNN collaboration requests
- [ ] Implement knowledge sharing via GNN
- [ ] Add specialist discovery via GNN
- [ ] Tests: Builder collaboration (8 tests)

**Day 24: Specialist Agent Integration**
- [ ] Add GNN capability broadcasting
- [ ] Implement task response routing
- [ ] Tests: Specialist communication (8 tests)

**Day 25: Risk Manager Integration**
- [ ] Add GNN risk warnings
- [ ] Implement team coordination via GNN
- [ ] Tests: Risk manager oversight (8 tests)

**Day 26: Comprehensive Testing**
- [ ] Integration tests (all components)
- [ ] Performance validation (overhead reduction)
- [ ] Scalability tests (100+ agents)
- [ ] Edge case testing

**Day 27: Documentation**
- [ ] API guide (similar to Big Rock 5)
- [ ] Usage examples
- [ ] Integration patterns
- [ ] Performance tuning guide

**Day 28: Final Validation & Delivery**
- [ ] Final test sweep (100% pass rate)
- [ ] Coverage report (target 85%+)
- [ ] Update PROGRESS.md
- [ ] Prepare for Big Rock 8

---

## Test Suite Design

### Test Categories (30+ cases)

**1. Graph Operations (15 tests)**
- Agent registration and removal
- Edge creation and removal
- Edge weight updates
- Neighbor queries
- Subgraph extraction
- Graph statistics
- Proximity-based edges
- Role-based edges
- Dynamic pruning
- Connectivity preservation

**2. Message Encoding (8 tests)**
- Message embedding generation
- Type encoding
- Priority encoding
- Content encoding
- Sender embedding integration
- Embedding consistency
- Dimension validation

**3. GNN Layers (12 tests)**
- Forward pass correctness
- Attention computation
- Multi-head attention
- Message aggregation (mean, sum, max)
- Embedding updates
- Layer chaining
- Gradient flow (if learning enabled)

**4. Message Propagation (10 tests)**
- Single-hop routing
- Multi-hop routing
- TTL enforcement
- Path tracking
- Relevance scoring
- Top-k selection
- Cycle prevention
- Targeted routing

**5. Communication API (15 tests)**
- Agent registration
- Send message (broadcast)
- Send message (targeted)
- Receive messages
- Message filtering
- Queue management
- Thread safety
- Message history

**6. Learning & Optimization (10 tests)**
- Edge weight updates from rewards
- Learning convergence
- Graph optimization
- Weak edge pruning
- Strong edge reinforcement
- Reward buffering

**7. Integration Tests (15 tests)**
- Integration with electrical signaling
- Integration with stigmergy
- Multi-layer coordination
- End-to-end scenarios (10+ agents)
- Collaboration request flow
- Knowledge sharing flow

**8. Performance Tests (10 tests)**
- Routing latency (< 5ms)
- Overhead reduction (40-60%)
- Scalability (100+ agents)
- Memory usage
- Concurrent message handling
- Graph optimization speed

**Total: 95 tests** (exceeds 30+ requirement)

---

## Risk Mitigation

### Risk 1: GNN Complexity
**Risk:** GNN implementation may be too complex for initial release
**Mitigation:**
- Start with simple attention mechanism (dot product)
- Use pre-computed embeddings (no backprop initially)
- Defer advanced features to future iterations

### Risk 2: Performance Overhead
**Risk:** GNN routing may add latency vs. simple broadcast
**Mitigation:**
- Implement fast path for critical signals (bypass GNN)
- Pre-compute frequently used routing patterns
- Batch message routing where possible

### Risk 3: Learning Instability
**Risk:** Edge weights may oscillate or converge poorly
**Mitigation:**
- Use exponential moving average for weight updates
- Add weight regularization (prevent extreme values)
- Implement graph stability metrics
- Allow manual edge seeding for critical paths

### Risk 4: Integration Complexity
**Risk:** Three communication layers (signals, stigmergy, GNN) may conflict
**Mitigation:**
- Clear priority: Signals > GNN > Stigmergy
- Complementary use cases (immediate vs. planned vs. indirect)
- Unified agent API abstracts complexity

---

## Success Criteria

1. **Overhead Reduction:** ≥ 40% reduction vs. broadcast in 50-agent scenarios
2. **Test Coverage:** 95+ tests, 100% pass rate, ≥ 85% code coverage
3. **Performance:** < 5ms routing latency for 90th percentile
4. **Scalability:** Support 100+ agents with < 100ms routing
5. **Integration:** Seamless integration with Big Rocks 4, 5, 6
6. **Documentation:** Comprehensive API guide with 5+ examples
7. **Learning:** Edge weights converge within 1000 messages

---

## Next Steps After Big Rock 7

**Phase 2 - Weeks 5-8:**
- Big Rock 8: Experience Replay Buffer
- Big Rock 9: Hierarchical RL Training
- Big Rock 10: Distributed Training Infrastructure

**Phase 3 - Weeks 9-12:**
- Big Rock 11: Evaluation Framework
- Big Rock 12: Production Deployment Tools

---

## References

1. Jiang, J., et al. (2022). "Graph Neural Networks for Multi-Agent Coordination"
2. Sukhbaatar, S., et al. (2016). "CommNet: Learning Multiagent Communication with Backpropagation"
3. Foerster, J., et al. (2016). "Learning to Communicate with Deep Multi-Agent Reinforcement Learning"
4. Veličković, P., et al. (2018). "Graph Attention Networks"
5. Battaglia, P., et al. (2018). "Relational Inductive Biases, Deep Learning, and Graph Networks"
6. Kipf, T., & Welling, M. (2017). "Semi-Supervised Classification with Graph Convolutional Networks"

---

**Status:** Planning Complete - Ready for Implementation
**Estimated LOC:** ~2,000 lines (core) + 300 lines (integration) + 1,500 lines (tests)
**Estimated Complexity:** High (GNN algorithms + graph optimization)
**Estimated Time:** 10 working days (Weeks 3-4)
