import os
import math
import json
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from ..base import BaseMemory, MemoryItem, MemoryConfig
from ..embedding import get_text_embedder
from ..storage.document_store import SQLiteDocumentStore
from ..storage.qdrant_store import QdrantConnectionManager

class Episode:
    def __init__(
        self,
        episode_id: str,
        user_id: str,
        session_id: str,
        timestamp: datetime,
        content: str,
        context: Dict[str, Any],
        outcome: Optional[str] = None,
        importance: float = 0.5
    ):
        self.episode_id = episode_id
        self.user_id = user_id
        self.session_id = session_id
        self.timestamp = timestamp
        self.content = content
        self.context = context
        self.outcome = outcome
        self.importance = importance

class EpisodicMemory(BaseMemory):
    def __init__(self, config: MemoryConfig, storage_backend=None):
        super().__init__(config, storage_backend)
        os.makedirs(self.config.storage_path, exist_ok=True)
        db_path = os.path.join(self.config.storage_path, "memory.db")
        self.doc_store = SQLiteDocumentStore(db_path=db_path)
        self.vector_store = QdrantConnectionManager.get_instance()
        self.embedder = get_text_embedder()
        self.episodes: List[Episode] = []
        self.sessions: Dict[str, List[str]] = {}
        self.patterns_cache = {}
        self.last_pattern_analysis = None

    def add(self, memory_item: MemoryItem) -> str:
        session_id = memory_item.metadata.get("session_id", "default_session")
        context = memory_item.metadata.get("context", {})
        outcome = memory_item.metadata.get("outcome")
        participants = memory_item.metadata.get("participants", [])
        tags = memory_item.metadata.get("tags", [])
        episode = Episode(
            episode_id=memory_item.id,
            user_id=memory_item.user_id,
            session_id=session_id,
            timestamp=memory_item.timestamp,
            content=memory_item.content,
            context=context,
            outcome=outcome,
            importance=memory_item.importance
        )
        self.episodes.append(episode)
        if session_id not in self.sessions:
            self.sessions[session_id] = []
        self.sessions[session_id].append(episode.episode_id)
        ts_int = int(memory_item.timestamp.timestamp())
        self.doc_store.add_memory(
            memory_id=memory_item.id,
            user_id=memory_item.user_id,
            content=memory_item.content,
            memory_type="episodic",
            timestamp=ts_int,
            importance=memory_item.importance,
            properties={
                "session_id": session_id,
                "context": context,
                "outcome": outcome,
                "participants": participants,
                "tags": tags
            }
        )
        embedding = self.embedder.encode(memory_item.content)
        if hasattr(embedding, "tolist"):
            embedding = embedding.tolist()
        self.vector_store.add_vectors(
            vectors=[embedding],
            metadata=[{
                "memory_id": memory_item.id,
                "user_id": memory_item.user_id,
                "memory_type": "episodic",
                "importance": memory_item.importance,
                "session_id": session_id,
                "content": memory_item.content
            }],
            ids=[memory_item.id]
        )
        return memory_item.id
    
    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
        user_id = kwargs.get("user_id")
        session_id = kwargs.get("session_id")
        time_range: Optional[Tuple[datetime, datetime]] = kwargs.get("time_range")
        importance_threshold: Optional[float] = kwargs.get("importance_threshold")
        candidate_ids: Optional[set] = None
        if time_range is not None or importance_threshold is not None:
            start_ts = int(time_range[0].timestamp()) if time_range else None
            end_ts = int(time_range[1].timestamp()) if time_range else None
            docs = self.doc_store.search_memories(
                user_id=user_id,
                memory_type="episodic",
                start_time=start_ts,
                end_time=end_ts,
                importance_threshold=importance_threshold,
                limit=1000
            )
            candidate_ids = {d["memory_id"] for d in docs}
        query_vec = self.embedder.encode(query)
        if hasattr(query_vec, "tolist"):
            query_vec = query_vec.tolist()
        where = {"memory_type": "episodic", "user_id": user_id}
        hits = self.vector_store.search_similar(
            query_vector=query_vec,
            limit=max(limit * 5, 20),
            where=where
        )
        now_ts = int(datetime.now().timestamp())
        results: List[Tuple[float, MemoryItem]] = []
        seen = set() 
        for hit in hits:
            meta = hit.get("metadata", {})
            mem_id = meta.get("memory_id")
            if not mem_id or mem_id in seen:
                continue
            episode = next((e for e in self.episodes if e.episode_id == mem_id), None)
            if episode and episode.context.get("forgotten", False):
                continue
            if candidate_ids is not None and mem_id not in candidate_ids:
                continue
            if session_id and meta.get("session_id") != session_id:
                continue
            doc = self.doc_store.get_memory(mem_id)
            if not doc:
                continue
            vec_score = float(hit.get("score", 0.0))
            age_days = max(0.0, (now_ts - int(doc["timestamp"])) / 86400.0)
            recency_score = 1.0 / (1.0 + age_days)
            imp = float(doc.get("importance", 0.5))
            base_relevance = vec_score * 0.8 + recency_score * 0.2
            importance_weight = 0.8 + (imp * 0.4)
            combined = base_relevance * importance_weight
            item = MemoryItem(
                id=doc["memory_id"],
                content=doc["content"],
                memory_type=doc["memory_type"],
                user_id=doc["user_id"],
                timestamp=datetime.fromtimestamp(doc["timestamp"]),
                importance=doc.get("importance", 0.5),
                metadata={
                    **doc.get("properties", {}),
                    "relevance_score": combined,
                    "vector_score": vec_score,
                    "recency_score": recency_score
                }
            )
            results.append((combined, item))
            seen.add(mem_id)
        if not results:
            fallback = super()._generate_id
            query_lower = query.lower()
            for ep in self._filter_episodes(user_id, session_id, time_range):
                if query_lower in ep.content.lower():
                    recency_score = 1.0 / (1.0 + max(0.0, (now_ts - int(ep.timestamp.timestamp())) / 86400.0))
                    keyword_score = 0.5
                    base_relevance = keyword_score * 0.8 + recency_score * 0.2
                    importance_weight = 0.8 + (ep.importance * 0.4)
                    combined = base_relevance * importance_weight
                    item = MemoryItem(
                        id=ep.episode_id,
                        content=ep.content,
                        memory_type="episodic",
                        user_id=ep.user_id,
                        timestamp=ep.timestamp,
                        importance=ep.importance,
                        metadata={
                            "session_id": ep.session_id,
                            "context": ep.context,
                            "outcome": ep.outcome,
                            "relevance_score": combined
                        }
                    )
                    results.append((combined, item))
        results.sort(key=lambda x: x[0], reverse=True)
        return [it for _, it in results[:limit]]
    
    def update(
        self,
        memory_id: str,
        content: str = None,
        importance: float = None,
        metadata: Dict[str, Any] = None
    ) -> bool:
        updated = False
        for episode in self.episodes:
            if episode.episode_id == memory_id:
                if content is not None:
                    episode.content = content
                if importance is not None:
                    episode.importance = importance
                if metadata is not None:
                    episode.context.update(metadata.get("context", {}))
                    if "outcome" in metadata:
                        episode.outcome = metadata["outcome"]
                updated = True
                break
        doc_updated = self.doc_store.update_memory(
            memory_id=memory_id,
            content=content,
            importance=importance,
            properties=metadata
        )
        if content is not None:
            embedding = self.embedder.encode(content)
            if hasattr(embedding, "tolist"):
                embedding = embedding.tolist()
            doc = self.doc_store.get_memory(memory_id)
            payload = {
                "memory_id": memory_id,
                "user_id": doc["user_id"] if doc else "",
                "memory_type": "episodic",
                "importance": (doc.get("importance") if doc else importance) or 0.5,
                "session_id": (doc.get("properties", {}) or {}).get("session_id"),
                "content": content
            }
            self.vector_store.add_vectors(
                vectors=[embedding],
                metadata=[payload],
                ids=[memory_id]
            )
        return updated or doc_updated
    
    def remove(self, memory_id: str) -> bool:
        removed = False
        for i, episode in enumerate(self.episodes):
            if episode.episode_id == memory_id:
                removed_episode = self.episodes.pop(i)
                session_id = removed_episode.session_id
                if session_id in self.sessions:
                    self.sessions[session_id].remove(memory_id)
                    if not self.sessions[session_id]:
                        del self.sessions[session_id]
                removed = True
                break
        doc_deleted = self.doc_store.delete_memory(memory_id)
        self.vector_store.delete_memories([memory_id])
        return removed or doc_deleted
    
    def has_memory(self, memory_id: str) -> bool:
        return any(episode.episode_id == memory_id for episode in self.episodes)
    
    def clear(self):
        self.episodes.clear()
        self.sessions.clear()
        self.patterns_cache.clear()
        docs = self.doc_store.search_memories(memory_type="episodic", limit=10000)
        ids = [d["memory_id"] for d in docs]
        for mid in ids:
            self.doc_store.delete_memory(mid)
        if ids:
            self.vector_store.delete_memories(ids)

    def forget(self, strategy: str = "importance_based", threshold: float = 0.1, max_age_days: int = 30) -> int:
        forgotten_count = 0
        current_time = datetime.now()
        to_remove = []   
        for episode in self.episodes:
            should_forget = False  
            if strategy == "importance_based":
                if episode.importance < threshold:
                    should_forget = True
            elif strategy == "time_based":
                cutoff_time = current_time - timedelta(days=max_age_days)
                if episode.timestamp < cutoff_time:
                    should_forget = True
            elif strategy == "capacity_based":
                if len(self.episodes) > self.config.max_capacity:
                    sorted_episodes = sorted(self.episodes, key=lambda e: e.importance)
                    excess_count = len(self.episodes) - self.config.max_capacity
                    if episode in sorted_episodes[:excess_count]:
                        should_forget = True         
            if should_forget:
                to_remove.append(episode.episode_id)
        for episode_id in to_remove:
            if self.remove(episode_id):
                forgotten_count += 1
        return forgotten_count

    def get_all(self) -> List[MemoryItem]:
        memory_items = []
        for episode in self.episodes:
            memory_item = MemoryItem(
                id=episode.episode_id,
                content=episode.content,
                memory_type="episodic",
                user_id=episode.user_id,
                timestamp=episode.timestamp,
                importance=episode.importance,
                metadata=episode.metadata
            )
            memory_items.append(memory_item)
        return memory_items
    
    def get_stats(self) -> Dict[str, Any]:
        active_episodes = self.episodes
        db_stats = self.doc_store.get_database_stats()
        vs_stats = self.vector_store.get_collection_stats()
        return {
            "count": len(active_episodes),
            "forgotten_count": 0,
            "total_count": len(self.episodes),
            "sessions_count": len(self.sessions),
            "avg_importance": sum(e.importance for e in active_episodes) / len(active_episodes) if active_episodes else 0.0,
            "time_span_days": self._calculate_time_span(),
            "memory_type": "episodic",
            "vector_store": vs_stats,
            "document_store": {k: v for k, v in db_stats.items() if k.endswith("_count") or k in ["store_type", "db_path"]}
        }
    
    def get_session_episodes(self, session_id: str) -> List[Episode]:
        if session_id not in self.sessions:
            return []
        episode_ids = self.sessions[session_id]
        return [e for e in self.episodes if e.episode_id in episode_ids]
    
    def find_patterns(self, user_id: str = None, min_frequency: int = 2) -> List[Dict[str, Any]]:
        cache_key = f"{user_id}_{min_frequency}"
        if (cache_key in self.patterns_cache and 
            self.last_pattern_analysis and 
            (datetime.now() - self.last_pattern_analysis).hours < 1):
            return self.patterns_cache[cache_key]
        episodes = [e for e in self.episodes if user_id is None or e.user_id == user_id]
        keyword_patterns = {}
        context_patterns = {}
        for episode in episodes:
            words = episode.content.lower().split()
            for word in words:
                if len(word) > 3:
                    keyword_patterns[word] = keyword_patterns.get(word, 0) + 1
            for key, value in episode.context.items():
                pattern_key = f"{key}:{value}"
                context_patterns[pattern_key] = context_patterns.get(pattern_key, 0) + 1
        patterns = []    
        for keyword, frequency in keyword_patterns.items():
            if frequency >= min_frequency:
                patterns.append({
                    "type": "keyword",
                    "pattern": keyword,
                    "frequency": frequency,
                    "confidence": frequency / len(episodes)
                })
        for context_pattern, frequency in context_patterns.items():
            if frequency >= min_frequency:
                patterns.append({
                    "type": "context",
                    "pattern": context_pattern,
                    "frequency": frequency,
                    "confidence": frequency / len(episodes)
                })
        patterns.sort(key=lambda x: x["frequency"], reverse=True)
        self.patterns_cache[cache_key] = patterns
        self.last_pattern_analysis = datetime.now()
        return patterns
    
    def get_timeline(self, user_id: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        episodes = [e for e in self.episodes if user_id is None or e.user_id == user_id]
        episodes.sort(key=lambda x: x.timestamp, reverse=True)
        timeline = []
        for episode in episodes[:limit]:
            timeline.append({
                "episode_id": episode.episode_id,
                "timestamp": episode.timestamp.isoformat(),
                "content": episode.content[:100] + "..." if len(episode.content) > 100 else episode.content,
                "session_id": episode.session_id,
                "importance": episode.importance,
                "outcome": episode.outcome
            })
        return timeline
    
    def _filter_episodes(
        self,
        user_id: str = None,
        session_id: str = None,
        time_range: Tuple[datetime, datetime] = None
    ) -> List[Episode]:
        filtered = self.episodes 
        if user_id:
            filtered = [e for e in filtered if e.user_id == user_id] 
        if session_id:
            filtered = [e for e in filtered if e.session_id == session_id] 
        if time_range:
            start_time, end_time = time_range
            filtered = [e for e in filtered if start_time <= e.timestamp <= end_time]     
        return filtered
    
    def _calculate_time_span(self) -> float:
        if not self.episodes:
            return 0.0
        timestamps = [e.timestamp for e in self.episodes]
        min_time = min(timestamps)
        max_time = max(timestamps)
        return (max_time - min_time).days
    
    def _persist_episode(self, episode: Episode):
        if self.storage and hasattr(self.storage, 'add_memory'):
            self.storage.add_memory(
                memory_id=episode.episode_id,
                user_id=episode.user_id,
                content=episode.content,
                memory_type="episodic",
                timestamp=int(episode.timestamp.timestamp()),
                importance=episode.importance,
                properties={
                    "session_id": episode.session_id,
                    "context": episode.context,
                    "outcome": episode.outcome
                }
            )

    def _remove_from_storage(self, memory_id: str):
        if self.storage and hasattr(self.storage, 'delete_memory'):
            self.storage.delete_memory(memory_id)