# ================================================================
#  TIMELINE ENGINE - Advanced Temporal Processing System
#  Author: Morpheus (Creator), Somnus Development Collective 
#  License: Proprietary Software License Agreement (Somnus Development Collective)
# ================================================================
import asyncio
import threading
import time
from collections import deque, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Set, Tuple
from enum import Enum
import heapq
import logging
import numpy as np

logger = logging.getLogger("TimelineEngine")

class EventPriority(Enum):
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4

@dataclass
class TemporalEvent:
    event_id: str
    timestamp: float
    event_type: str
    payload: Dict[str, Any]
    priority: EventPriority = EventPriority.NORMAL
    causality_chain: List[str] = field(default_factory=list)
    processed: bool = False
    retry_count: int = 0
    max_retries: int = 3
    
    def __lt__(self, other):
        return (self.priority.value, self.timestamp) < (other.priority.value, other.timestamp)

@dataclass
class TimelineMetrics:
    total_events_processed: int = 0
    events_per_second: float = 0.0
    average_latency: float = 0.0
    queue_depth: int = 0
    batch_efficiency: float = 0.0
    memory_usage_mb: float = 0.0
    cache_hit_rate: float = 0.0

class DynamicBatchProcessor:
    """Advanced batching system with adaptive sizing and prefetching"""
    
    def __init__(self, max_batch_size: int = 32, target_latency_ms: float = 10.0):
        self.max_batch_size = max_batch_size
        self.target_latency_ms = target_latency_ms
        self.batch_history = deque(maxlen=100)
        self.optimal_batch_size = 8
        self.prefetch_buffer = deque(maxlen=64)
        
    def calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on throughput and latency metrics"""
        if len(self.batch_history) < 5:
            return self.optimal_batch_size
            
        recent_batches = list(self.batch_history)[-20:]
        throughputs = []
        latencies = []
        
        for batch_size, processing_time, items_processed in recent_batches:
            throughput = items_processed / processing_time if processing_time > 0 else 0
            latency = processing_time * 1000  # Convert to ms
            throughputs.append((batch_size, throughput))
            latencies.append((batch_size, latency))
        
        # Find batch size with best throughput within latency constraints
        valid_sizes = [size for size, lat in latencies if lat <= self.target_latency_ms]
        if not valid_sizes:
            valid_sizes = [size for size, _ in throughputs]
        
        if valid_sizes:
            best_throughput = max((thr for size, thr in throughputs if size in valid_sizes))
            self.optimal_batch_size = next(size for size, thr in throughputs if thr == best_throughput)
        
        return min(self.optimal_batch_size, self.max_batch_size)
    
    def should_flush_batch(self, current_batch_size: int, time_since_last_flush: float) -> bool:
        """Determine if current batch should be flushed based on adaptive criteria"""
        optimal_size = self.calculate_optimal_batch_size()
        
        # Flush if we've reached optimal size
        if current_batch_size >= optimal_size:
            return True
        
        # Flush if we've waited too long (latency constraint)
        if time_since_last_flush * 1000 > self.target_latency_ms * 0.8:
            return True
        
        # Flush if we have high priority events
        return False
    
    def record_batch_performance(self, batch_size: int, processing_time: float, items_processed: int):
        """Record batch performance metrics for optimization"""
        self.batch_history.append((batch_size, processing_time, items_processed))

class TemporalKVCache:
    """Multi-level KV cache for temporal data with intelligent prefetching"""
    
    def __init__(self, l1_size: int = 1000, l2_size: int = 10000, ttl_seconds: float = 300.0):
        self.l1_cache = {}  # Hot cache
        self.l2_cache = {}  # Cold cache
        self.l1_size = l1_size
        self.l2_size = l2_size
        self.ttl_seconds = ttl_seconds
        self.access_counts = defaultdict(int)
        self.access_times = {}
        self.prefetch_queue = deque()
        
    def get(self, key: str) -> Optional[Any]:
        """Get value with promotion between cache levels"""
        current_time = time.time()
        
        # Check L1 cache first
        if key in self.l1_cache:
            value, timestamp = self.l1_cache[key]
            if current_time - timestamp < self.ttl_seconds:
                self.access_counts[key] += 1
                self.access_times[key] = current_time
                return value
            else:
                del self.l1_cache[key]
        
        # Check L2 cache
        if key in self.l2_cache:
            value, timestamp = self.l2_cache[key]
            if current_time - timestamp < self.ttl_seconds:
                self.access_counts[key] += 1
                self.access_times[key] = current_time
                # Promote to L1 if frequently accessed
                if self.access_counts[key] > 3:
                    self._promote_to_l1(key, value, current_time)
                return value
            else:
                del self.l2_cache[key]
        
        return None
    
    def put(self, key: str, value: Any):
        """Put value in cache with intelligent placement"""
        current_time = time.time()
        
        # Evict if necessary
        self._evict_expired(current_time)
        
        # Decide cache level based on access patterns
        if self.access_counts[key] > 2 or key in self.prefetch_queue:
            self._put_l1(key, value, current_time)
        else:
            self._put_l2(key, value, current_time)
    
    def _promote_to_l1(self, key: str, value: Any, timestamp: float):
        """Promote item from L2 to L1"""
        if key in self.l2_cache:
            del self.l2_cache[key]
        self._put_l1(key, value, timestamp)
    
    def _put_l1(self, key: str, value: Any, timestamp: float):
        """Put item in L1 cache with LRU eviction"""
        if len(self.l1_cache) >= self.l1_size and key not in self.l1_cache:
            # Evict LRU item
            lru_key = min(self.l1_cache.keys(), 
                         key=lambda k: self.access_times.get(k, 0))
            del self.l1_cache[lru_key]
        
        self.l1_cache[key] = (value, timestamp)
    
    def _put_l2(self, key: str, value: Any, timestamp: float):
        """Put item in L2 cache with LRU eviction"""
        if len(self.l2_cache) >= self.l2_size and key not in self.l2_cache:
            # Evict LRU item
            lru_key = min(self.l2_cache.keys(), 
                         key=lambda k: self.access_times.get(k, 0))
            del self.l2_cache[lru_key]
        
        self.l2_cache[key] = (value, timestamp)
    
    def _evict_expired(self, current_time: float):
        """Evict expired items from both cache levels"""
        expired_l1 = [k for k, (_, ts) in self.l1_cache.items() 
                      if current_time - ts > self.ttl_seconds]
        expired_l2 = [k for k, (_, ts) in self.l2_cache.items() 
                      if current_time - ts > self.ttl_seconds]
        
        for key in expired_l1:
            del self.l1_cache[key]
        for key in expired_l2:
            del self.l2_cache[key]
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = sum(self.access_counts.values())
        l1_items = len(self.l1_cache)
        l2_items = len(self.l2_cache)
        
        return {
            "l1_size": l1_items,
            "l2_size": l2_items,
            "total_requests": total_requests,
            "l1_utilization": l1_items / self.l1_size,
            "l2_utilization": l2_items / self.l2_size
        }

class TimelineEngine:
    """Advanced timeline processing engine with dynamic batching and caching"""
    
    def __init__(self, max_workers: int = 4, enable_prefetching: bool = True):
        self.event_queue = []
        self.processed_events = {}
        self.event_handlers = {}
        self.batch_processor = DynamicBatchProcessor()
        self.kv_cache = TemporalKVCache()
        self.metrics = TimelineMetrics()
        
        self.max_workers = max_workers
        self.enable_prefetching = enable_prefetching
        self.worker_pool = []
        self.is_running = False
        self.processing_lock = threading.Lock()
        
        self._setup_worker_pool()
        
    def _setup_worker_pool(self):
        """Setup worker thread pool for event processing"""
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                name=f"TimelineWorker-{i}",
                daemon=True
            )
            self.worker_pool.append(worker)
    
    def start(self):
        """Start the timeline engine"""
        if self.is_running:
            return
        
        self.is_running = True
        for worker in self.worker_pool:
            worker.start()
        
        logger.info("Timeline engine started with {} workers".format(self.max_workers))
    
    def stop(self):
        """Stop the timeline engine"""
        self.is_running = False
        for worker in self.worker_pool:
            worker.join(timeout=5.0)
        
        logger.info("Timeline engine stopped")
    
    def register_event_handler(self, event_type: str, handler: Callable[[TemporalEvent], Any]):
        """Register handler for specific event type"""
        if event_type not in self.event_handlers:
            self.event_handlers[event_type] = []
        self.event_handlers[event_type].append(handler)
    
    def schedule_event(self, event: TemporalEvent):
        """Schedule event for processing"""
        with self.processing_lock:
            heapq.heappush(self.event_queue, event)
            self.metrics.queue_depth = len(self.event_queue)
    
    def _worker_loop(self):
        """Main worker loop for event processing"""
        batch = []
        last_flush_time = time.time()
        
        while self.is_running:
            try:
                current_time = time.time()
                
                # Collect events for batch processing
                while (len(batch) < self.batch_processor.max_batch_size and
                       self.event_queue):
                    
                    with self.processing_lock:
                        if self.event_queue:
                            event = heapq.heappop(self.event_queue)
                            if event.timestamp <= current_time:
                                batch.append(event)
                            else:
                                # Event is scheduled for future, put back
                                heapq.heappush(self.event_queue, event)
                                break
                
                # Determine if we should flush the batch
                should_flush = (
                    batch and
                    self.batch_processor.should_flush_batch(
                        len(batch), 
                        current_time - last_flush_time
                    )
                )
                
                if should_flush:
                    self._process_batch(batch)
                    batch = []
                    last_flush_time = current_time
                else:
                    # Small sleep to prevent busy waiting
                    time.sleep(0.001)
                    
            except Exception as e:
                logger.error(f"Error in worker loop: {e}", exc_info=True)
                time.sleep(0.1)
    
    def _process_batch(self, batch: List[TemporalEvent]):
        """Process a batch of events with performance tracking"""
        if not batch:
            return
        
        start_time = time.time()
        processed_count = 0
        
        try:
            # Sort batch by priority and timestamp
            batch.sort()
            
            for event in batch:
                if self._process_single_event(event):
                    processed_count += 1
            
            # Record batch performance
            processing_time = time.time() - start_time
            self.batch_processor.record_batch_performance(
                len(batch), processing_time, processed_count
            )
            
            # Update metrics
            self.metrics.total_events_processed += processed_count
            self.metrics.events_per_second = processed_count / processing_time if processing_time > 0 else 0
            self.metrics.average_latency = processing_time / len(batch) if batch else 0
            
        except Exception as e:
            logger.error(f"Error processing batch: {e}", exc_info=True)
    
    def _process_single_event(self, event: TemporalEvent) -> bool:
        """Process a single event with caching and error handling"""
        try:
            # Check cache first
            cache_key = f"{event.event_type}:{event.event_id}"
            cached_result = self.kv_cache.get(cache_key)
            
            if cached_result is not None:
                logger.debug(f"Cache hit for event {event.event_id}")
                return True
            
            # Process event with registered handlers
            handlers = self.event_handlers.get(event.event_type, [])
            if not handlers:
                logger.warning(f"No handlers registered for event type: {event.event_type}")
                return False
            
            results = []
            for handler in handlers:
                try:
                    result = handler(event)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Handler error for event {event.event_id}: {e}")
                    if event.retry_count < event.max_retries:
                        event.retry_count += 1
                        self.schedule_event(event)
                        return False
            
            # Cache successful result
            self.kv_cache.put(cache_key, results)
            self.processed_events[event.event_id] = event
            event.processed = True
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing event {event.event_id}: {e}", exc_info=True)
            return False
    
    def get_metrics(self) -> TimelineMetrics:
        """Get current performance metrics"""
        self.metrics.queue_depth = len(self.event_queue)
        
        cache_stats = self.kv_cache.get_cache_stats()
        total_cache_requests = cache_stats["total_requests"]
        if total_cache_requests > 0:
            self.metrics.cache_hit_rate = (
                cache_stats["l1_size"] + cache_stats["l2_size"]
            ) / total_cache_requests
        
        self.metrics.batch_efficiency = self.batch_processor.optimal_batch_size / self.batch_processor.max_batch_size
        
        return self.metrics