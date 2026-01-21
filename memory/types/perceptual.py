import os
import random
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Union, Tuple
from ..base import BaseMemory, MemoryItem, MemoryConfig
from ..embedding import get_text_embedder, get_dimension
from ..storage.document_store import SQLiteDocumentStore
from ..storage.qdrant_store import QdrantVectorStore, QdrantConnectionManager

class Perception:
    def __init__(
        self,
        perception_id: str,
        data: Any,
        modality: str,
        encoding: Optional[List[float]] = None,
        metadata: Dict[str, Any] = None
    ):
        self.perception_id = perception_id
        self.data = data
        self.modality = modality
        self.encoding = encoding or []
        self.metadata = metadata or {}
        self.timestamp = datetime.now()
        self.data_hash = self._calculate_hash()
    
    def _calculate_hash(self) -> str:
        if isinstance(self.data, str):
            return hashlib.md5(self.data.encode()).hexdigest()
        elif isinstance(self.data, bytes):
            return hashlib.md5(self.data).hexdigest()
        else:
            return hashlib.md5(str(self.data).encode()).hexdigest()

class PerceptualMemory(BaseMemory):
    def __init__(self, config: MemoryConfig, storage_backend=None):
        super().__init__(config, storage_backend)
        os.makedirs(self.config.storage_path, exist_ok=True)
        db_path = os.path.join(self.config.storage_path, "memory.db")
        self.doc_store = SQLiteDocumentStore(db_path=db_path)
        self.text_embedder = get_text_embedder()
        self.vector_dim = get_dimension(getattr(self.text_embedder, "dimension", 384))
        self.perceptions: Dict[str, Perception] = {}
        self.perceptual_memories: List[MemoryItem] = []
        self.modality_index: Dict[str, List[str]] = {}
        self.supported_modalities = set(self.config.perceptual_memory_modalities)
        try:
            from transformers import CLIPModel, CLIPProcessor
            self._clip_model = CLIPModel.from_pretrained(self.config.perceptual_memory_clip_mpdel)
            self._clip_processor = CLIPProcessor.from_pretrained(self.config.perceptual_memory_clip_mpdel)
            self._image_dim = self._clip_model.config.projection_dim if hasattr(self._clip_model.config, "projection_dim") else 512
        except Exception:
            self._clip_model = None
            self._clip_processor = None
            self._image_dim = self.vector_dim
        try:
            from transformers import ClapProcessor, ClapModel
            self._clap_model = ClapModel.from_pretrained(self.config.perceptual_memory_clap_mpdel)
            self._clap_processor = ClapProcessor.from_pretrained(self.config.perceptual_memory_clap_mpdel)
            self._audio_dim = getattr(self._clap_model.config, "projection_dim", None) or 512
        except Exception:
            self._clap_model = None
            self._clap_processor = None
            self._audio_dim = self.vector_dim
        self.vector_stores: Dict[str, QdrantVectorStore] = {}
        self.vector_stores["text"] = QdrantConnectionManager.get_instance(
            collection_name=f"hello_agents_vectors_text",
            vector_size=self.vector_dim
        )
        self.vector_stores["image"] = QdrantConnectionManager.get_instance(
            collection_name=f"hello_agents_vectors_image",
            vector_size=self._image_dim
        )
        self.vector_stores["audio"] = QdrantConnectionManager.get_instance(
            collection_name=f"hello_agents_vectors_audio",
            vector_size=self._audio_dim
        )
        self.encoders = self._init_encoders()
    
    def add(self, memory_item: MemoryItem) -> str:
        modality = memory_item.metadata.get("modality", "text")
        raw_data = memory_item.metadata.get("raw_data", memory_item.content)
        if modality not in self.supported_modalities:
            raise ValueError(f"不支持的模态类型: {modality}")
        perception = self._encode_perception(raw_data, modality, memory_item.id)
        self.perceptions[perception.perception_id] = perception
        if modality not in self.modality_index:
            self.modality_index[modality] = []
        self.modality_index[modality].append(perception.perception_id)
        memory_item.metadata["perception_id"] = perception.perception_id
        memory_item.metadata["modality"] = modality
        self.perceptual_memories.append(memory_item)
        self.doc_store.add_memory(
            memory_id=memory_item.id,
            user_id=memory_item.user_id,
            content=memory_item.content,
            memory_type="perceptual",
            timestamp=int(memory_item.timestamp.timestamp()),
            importance=memory_item.importance,
            properties={
                "perception_id": perception.perception_id,
                "modality": modality,
                "context": memory_item.metadata.get("context", {}),
                "tags": memory_item.metadata.get("tags", []),
            }
        )
        vector = perception.encoding
        store = self._get_vector_store_for_modality(modality)
        store.add_vectors(
            vectors=[vector],
            metadata=[{
                "memory_id": memory_item.id,
                "user_id": memory_item.user_id,
                "memory_type": "perceptual",
                "modality": modality,
                "importance": memory_item.importance,
                "content": memory_item.content,
            }],
            ids=[memory_item.id]
        )
        return memory_item.id
    
    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
        user_id = kwargs.get("user_id")
        target_modality = kwargs.get("target_modality")
        query_modality = kwargs.get("query_modality", target_modality or "text")
        qvec = self._encode_data(query, query_modality)
        where = {"memory_type": "perceptual"}
        if user_id:
            where["user_id"] = user_id
        if target_modality:
            where["modality"] = target_modality
        store = self._get_vector_store_for_modality(target_modality or query_modality)
        hits = store.search_similar(
            query_vector=qvec,
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
            if target_modality and meta.get("modality") != target_modality:
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
                importance=imp,
                metadata={**doc.get("properties", {}), "relevance_score": combined,
                          "vector_score": vec_score, "recency_score": recency_score}
            )
            results.append((combined, item))
            seen.add(mem_id)
        if not results:
            for m in self.perceptual_memories:
                if target_modality and m.metadata.get("modality") != target_modality:
                    continue
                if query.lower() in (m.content or "").lower():
                    recency_score = 1.0 / (1.0 + max(0.0, (now_ts - int(m.timestamp.timestamp())) / 86400.0))
                    keyword_score = 0.5
                    base_relevance = keyword_score * 0.8 + recency_score * 0.2
                    importance_weight = 0.8 + (m.importance * 0.4)
                    combined = base_relevance * importance_weight
                    results.append((combined, m))
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
        modality_cache = None
        for memory in self.perceptual_memories:
            if memory.id == memory_id:
                if content is not None:
                    memory.content = content
                if importance is not None:
                    memory.importance = importance
                if metadata is not None:
                    memory.metadata.update(metadata)
                modality_cache = memory.metadata.get("modality", "text")
                updated = True
                break
        self.doc_store.update_memory(
            memory_id=memory_id,
            content=content,
            importance=importance,
            properties=metadata
        )
        if content is not None or (metadata and "raw_data" in metadata):
            modality = metadata.get("modality", modality_cache or "text") if metadata else (modality_cache or "text")
            raw = metadata.get("raw_data", content) if metadata else content
            perception = self._encode_perception(raw or "", modality, memory_id)
            payload = self.doc_store.get_memory(memory_id) or {}
            self.vector_store.add_vectors(
                vectors=[perception.encoding],
                metadata=[{
                    "memory_id": memory_id,
                    "user_id": payload.get("user_id", ""),
                    "memory_type": "perceptual",
                    "modality": modality,
                    "importance": (payload.get("importance") or importance) or 0.5,
                    "content": content or (payload.get("content", "")),
                }],
                ids=[memory_id]
            )
        return updated
    
    def remove(self, memory_id: str) -> bool:
        removed = False
        for i, memory in enumerate(self.perceptual_memories):
            if memory.id == memory_id:
                removed_memory = self.perceptual_memories.pop(i)
                perception_id = removed_memory.metadata.get("perception_id")
                if perception_id and perception_id in self.perceptions:
                    perception = self.perceptions.pop(perception_id)
                    modality = perception.modality
                    if modality in self.modality_index:
                        if perception_id in self.modality_index[modality]:
                            self.modality_index[modality].remove(perception_id)
                        if not self.modality_index[modality]:
                            del self.modality_index[modality]
                removed = True
                break
        self.doc_store.delete_memory(memory_id)
        for store in self.vector_stores.values():
            store.delete_memories([memory_id])
        return removed
    
    def has_memory(self, memory_id: str) -> bool:
        return any(memory.id == memory_id for memory in self.perceptual_memories)
    
    def forget(self, strategy: str = "importance_based", threshold: float = 0.1, max_age_days: int = 30) -> int:
        forgotten_count = 0
        current_time = datetime.now()
        to_remove = []
        for memory in self.perceptual_memories:
            should_forget = False 
            if strategy == "importance_based":
                if memory.importance < threshold:
                    should_forget = True
            elif strategy == "time_based":
                cutoff_time = current_time - timedelta(days=max_age_days)
                if memory.timestamp < cutoff_time:
                    should_forget = True
            elif strategy == "capacity_based":
                if len(self.perceptual_memories) > self.config.max_capacity:
                    sorted_memories = sorted(self.perceptual_memories, key=lambda m: m.importance)
                    excess_count = len(self.perceptual_memories) - self.config.max_capacity
                    if memory in sorted_memories[:excess_count]:
                        should_forget = True
            if should_forget:
                to_remove.append(memory.id)
        for memory_id in to_remove:
            if self.remove(memory_id):
                forgotten_count += 1 
        return forgotten_count

    def clear(self):
        self.perceptual_memories.clear()
        self.perceptions.clear()
        self.modality_index.clear()
        docs = self.doc_store.search_memories(memory_type="perceptual", limit=10000)
        ids = [d["memory_id"] for d in docs]
        for mid in ids:
            self.doc_store.delete_memory(mid)
        for store in self.vector_stores.values():
            if ids:
                store.delete_memories(ids)

    def get_all(self) -> List[MemoryItem]:
        return self.perceptual_memories.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        active_memories = self.perceptual_memories
        modality_counts = {modality: len(ids) for modality, ids in self.modality_index.items()}
        vs_stats_all = {}
        for mod, store in self.vector_stores.items():
            vs_stats_all[mod] = store.get_collection_stats()
        db_stats = self.doc_store.get_database_stats()
        return {
            "count": len(active_memories), 
            "forgotten_count": 0, 
            "total_count": len(self.perceptual_memories), 
            "perceptions_count": len(self.perceptions),
            "modality_counts": modality_counts,
            "supported_modalities": list(self.supported_modalities),
            "avg_importance": sum(m.importance for m in active_memories) / len(active_memories) if active_memories else 0.0,
            "memory_type": "perceptual",
            "vector_stores": vs_stats_all,
            "document_store": {k: v for k, v in db_stats.items() if k.endswith("_count") or k in ["store_type", "db_path"]}
        }
    
    def cross_modal_search(
        self,
        query: Any,
        query_modality: str,
        target_modality: str = None,
        limit: int = 5
    ) -> List[MemoryItem]:
        return self.retrieve(
            query=str(query),
            limit=limit,
            query_modality=query_modality,
            target_modality=target_modality
        )
    
    def get_by_modality(self, modality: str, limit: int = 10) -> List[MemoryItem]:
        if modality not in self.modality_index:
            return []
        perception_ids = self.modality_index[modality]
        results = []
        for memory in self.perceptual_memories:
            if memory.metadata.get("perception_id") in perception_ids:
                results.append(memory)
                if len(results) >= limit:
                    break
        return results
    
    def generate_content(self, prompt: str, target_modality: str) -> Optional[str]:
        if target_modality not in self.supported_modalities:
            return None
        relevant_memories = self.retrieve(prompt, limit=3)
        if not relevant_memories:
            return None
        if target_modality == "text":
            contents = [memory.content for memory in relevant_memories]
            return f"基于感知记忆生成的内容：\n" + "\n".join(contents)
        return f"生成的{target_modality}内容（基于{len(relevant_memories)}个相关记忆）"
    
    def _init_encoders(self) -> Dict[str, Any]:
        encoders = {}
        for modality in self.supported_modalities:
            if modality == "text":
                encoders[modality] = self._text_encoder
            elif modality == "image":
                encoders[modality] = self._image_encoder
            elif modality == "audio":
                encoders[modality] = self._audio_encoder
            else:
                encoders[modality] = self._default_encoder
        return encoders
    
    def _encode_perception(self, data: Any, modality: str, memory_id: str) -> Perception:
        encoding = self._encode_data(data, modality)
        perception = Perception(
            perception_id=f"perception_{memory_id}",
            data=data,
            modality=modality,
            encoding=encoding,
            metadata={"source": "memory_system"}
        )
        return perception
    
    def _encode_data(self, data: Any, modality: str) -> List[float]:
        target_dim = self._get_dim_for_modality(modality)
        encoder = self.encoders.get(modality, self._default_encoder)
        vec = encoder(data)
        if not isinstance(vec, list):
            vec = list(vec)
        if len(vec) < target_dim:
            vec = vec + [0.0] * (target_dim - len(vec))
        elif len(vec) > target_dim:
            vec = vec[:target_dim]
        return vec
    
    def _text_encoder(self, text: str) -> List[float]:
        emb = self.text_embedder.encode(text or "")
        if hasattr(emb, "tolist"):
            emb = emb.tolist()
        return emb
    
    def _image_encoder_hash(self, image_data: Any) -> List[float]:
        try:
            if isinstance(image_data, (bytes, bytearray)):
                data_bytes = bytes(image_data)
            elif isinstance(image_data, str) and os.path.exists(image_data):
                with open(image_data, "rb") as f:
                    data_bytes = f.read()
            else:
                data_bytes = str(image_data).encode("utf-8", errors="ignore")
            hex_str = hashlib.sha256(data_bytes).hexdigest()
            return self._hash_to_vector(hex_str, self._get_dim_for_modality("image"))
        except Exception:
            return self._hash_to_vector(str(image_data), self._get_dim_for_modality("image"))

    def _image_encoder(self, image_data: Any) -> List[float]:
        if self._clip_model is None or self._clip_processor is None:
            return self._image_encoder_hash(image_data)
        try:
            from PIL import Image
            if isinstance(image_data, str) and os.path.exists(image_data):
                image = Image.open(image_data).convert("RGB")
            elif isinstance(image_data, (bytes, bytearray)):
                from io import BytesIO
                image = Image.open(BytesIO(bytes(image_data))).convert("RGB")
            else:
                return self._image_encoder_hash(image_data)
            inputs = self._clip_processor(images=image, return_tensors="pt")
            with self._no_grad():
                feats = self._clip_model.get_image_features(**inputs)
            vec = feats[0].detach().cpu().numpy().tolist()
            return vec
        except Exception:
            return self._image_encoder_hash(image_data)
    
    def _audio_encoder_hash(self, audio_data: Any) -> List[float]:
        try:
            if isinstance(audio_data, (bytes, bytearray)):
                data_bytes = bytes(audio_data)
            elif isinstance(audio_data, str) and os.path.exists(audio_data):
                with open(audio_data, "rb") as f:
                    data_bytes = f.read()
            else:
                data_bytes = str(audio_data).encode("utf-8", errors="ignore")
            hex_str = hashlib.sha256(data_bytes).hexdigest()
            return self._hash_to_vector(hex_str, self._get_dim_for_modality("audio"))
        except Exception:
            return self._hash_to_vector(str(audio_data), self._get_dim_for_modality("audio"))

    def _audio_encoder(self, audio_data: Any) -> List[float]:
        if self._clap_model is None or self._clap_processor is None:
            return self._audio_encoder_hash(audio_data)
        try:
            import numpy as np
            import librosa
            if isinstance(audio_data, str) and os.path.exists(audio_data):
                speech, sr = librosa.load(audio_data, sr=48000, mono=True)
            elif isinstance(audio_data, (bytes, bytearray)):
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(bytes(audio_data))
                    tmp_path = tmp.name
                speech, sr = librosa.load(tmp_path, sr=48000, mono=True)
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
            else:
                return self._audio_encoder_hash(audio_data)
            inputs = self._clap_processor(audios=speech, sampling_rate=48000, return_tensors="pt")
            with self._no_grad():
                feats = self._clap_model.get_audio_features(**inputs)
            vec = feats[0].detach().cpu().numpy().tolist()
            return vec
        except Exception:
            return self._audio_encoder_hash(audio_data)

    def _default_encoder(self, data: Any) -> List[float]:
        try:
            return self._text_encoder(str(data))
        except Exception:
            return self._hash_to_vector(str(data), self.vector_dim)
    
    def _calculate_similarity(self, encoding1: List[float], encoding2: List[float]) -> float:
        if not encoding1 or not encoding2:
            return 0.0
        min_len = min(len(encoding1), len(encoding2))
        if min_len == 0:
            return 0.0
        dot_product = sum(a * b for a, b in zip(encoding1[:min_len], encoding2[:min_len]))
        norm1 = sum(a * a for a in encoding1[:min_len]) ** 0.5
        norm2 = sum(a * a for a in encoding2[:min_len]) ** 0.5
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)

    def _hash_to_vector(self, data_str: str, dim: int) -> List[float]:
        seed = int(hashlib.sha256(data_str.encode("utf-8", errors="ignore")).hexdigest(), 16) % (2**32)
        rng = random.Random(seed)
        return [rng.random() for _ in range(dim)]

    class _no_grad:
        def __enter__(self):
            try:
                import torch
                self.prev = torch.is_grad_enabled()
                torch.set_grad_enabled(False)
            except Exception:
                self.prev = None
            return self
        def __exit__(self, exc_type, exc, tb):
            try:
                import torch
                if self.prev is not None:
                    torch.set_grad_enabled(self.prev)
            except Exception:
                pass

    def _get_vector_store_for_modality(self, modality: Optional[str]) -> QdrantVectorStore:
        mod = (modality or "text").lower()
        return self.vector_stores.get(mod, self.vector_stores["text"])

    def _get_dim_for_modality(self, modality: Optional[str]) -> int:
        mod = (modality or "text").lower()
        if mod == "image":
            return int(self._image_dim or self.vector_dim)
        if mod == "audio":
            return int(self._audio_dim or self.vector_dim)
        return int(self.vector_dim)