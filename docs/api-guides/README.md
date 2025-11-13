# MAE v3 API Documentation

This directory contains comprehensive API guides for MAE v3 features.

## Available API Guides

### Communication & Signaling
- **[Big Rock 5 API: Electrical Signaling](BIG_ROCK_5_API_GUIDE.md)**
  - Signal types and priority levels
  - Subscription and callback patterns
  - Performance optimization
  - Integration examples

### Advanced Learning
- **[Big Rock 8 API: Transfer Learning](BIG_ROCK_8_API_GUIDE.md)**
  - Transfer learning methods
  - Meta-learning (MAML) API
  - Task management
  - Performance evaluation

### Memory Systems
- **[Big Rock 9 API: Episodic Memory](BIG_ROCK_9_API_GUIDE.md)**
  - Experience replay buffers
  - Priority sampling
  - Memory consolidation
  - Integration with learning loops

## API Design Principles

All MAE v3 APIs follow these principles:

1. **Pythonic**: Intuitive, idiomatic Python interfaces
2. **Type-Safe**: Full type hints for IDE support
3. **Async-First**: Non-blocking operations for performance
4. **Extensible**: Easy to customize and extend
5. **Well-Documented**: Comprehensive docstrings and examples

## Usage Patterns

### Basic Integration

```python
from src.agents.base_agent import MycelialAgent

class MyAgent(MycelialAgent):
    def __init__(self, ...):
        super().__init__(...)

        # Electrical signaling
        self.subscribe_to_signal(SignalType.DANGER, self.handle_danger)

        # Episodic memory
        self.enable_episodic_memory(buffer_size=10000)

        # Transfer learning
        self.enable_transfer_learning(meta_learning=True)
```

## Related Documentation

- **[Big Rock Plans](../big-rocks/)** - Implementation details
- **[Architecture](../../ARCHITECTURE.md)** - System architecture
- **[Examples](../../examples/)** - Complete working examples

---

**MAE v3 Production Release** - All APIs Complete ✅
