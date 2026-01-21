import json
import math
import spacy
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set, Tuple
from ..base import BaseMemory, MemoryItem, MemoryConfig
from ..embedding import get_text_embedder
from ..storage.qdrant_store import QdrantConnectionManager
from ..storage.neo4j_store import Neo4jGraphStore

class Entity:
    def __init__(
        self,
        entity_id: str,
        name: str,
        entity_type: str = "MISC",
        description: str = "",
        properties: Dict[str, Any] = None
    ):
        self.entity_id = entity_id
        self.name = name
        self.entity_type = entity_type
        self.description = description
        self.properties = properties or {}
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.frequency = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_id": self.entity_id,
            "name": self.name,
            "entity_type": self.entity_type,
            "description": self.description,
            "properties": self.properties,
            "frequency": self.frequency
        }

class Relation:
    def __init__(
        self,
        from_entity: str,
        to_entity: str,
        relation_type: str,
        strength: float = 1.0,
        evidence: str = "",
        properties: Dict[str, Any] = None
    ):
        self.from_entity = from_entity
        self.to_entity = to_entity
        self.relation_type = relation_type
        self.strength = strength
        self.evidence = evidence
        self.properties = properties or {}
        self.created_at = datetime.now()
        self.frequency = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "from_entity": self.from_entity,
            "to_entity": self.to_entity,
            "relation_type": self.relation_type,
            "strength": self.strength,
            "evidence": self.evidence,
            "properties": self.properties,
            "frequency": self.frequency
        }

class SemanticMemory(BaseMemory):
    def __init__(self, config: MemoryConfig, storage_backend=None):
        super().__init__(config, storage_backend)
        self.embedding_model = get_text_embedder()
        self.vector_store = QdrantConnectionManager.get_instance()
        self.graph_store = Neo4jGraphStore()
        self._database_heartbeat_check()
        self._init_nlp()
        self.entities: Dict[str, Entity] = {}
        self.relations: List[Relation] = []
        self.semantic_memories: List[MemoryItem] = []
        self.memory_embeddings: Dict[str, np.ndarray] = {}
          
    def _database_heartbeat_check(self):
        vector_check = self.vector_store.heartbeat_check()
        graph_check = self.graph_store.heartbeat_check()
        print(f"💓\x20数据库通过心跳检测：Qdrant={vector_check}, Neo4j={graph_check}")
    
    def _init_nlp(self):
        try:
            self.nlp_zh = spacy.load("zh_core_web_sm")
        except Exception as e:
            self.nlp_zh = None
            print(f"⚠️\x20\x20中文spaCy模型不可用：{str(e)}")
        try:
            self.nlp_en = spacy.load("en_core_web_sm")
        except Exception as e:
            self.nlp_en = None
            print(f"⚠️\x20\x20英文spaCy模型不可用：{str(e)}")
        if not any([self.nlp_zh, self.nlp_en]):
            print("⛔\x20中文和英文spaCy模型均不可用，实体提取功能将受限")
    
    def add(self, memory_item: MemoryItem) -> str:
        embedding = self.embedding_model.encode(memory_item.content)
        self.memory_embeddings[memory_item.id] = embedding
        entities = self._extract_entities(memory_item.content)
        relations = self._extract_relations(memory_item.content, entities)
        for entity in entities:
            self._add_entity_to_graph(entity, memory_item)
        for relation in relations:
            self._add_relation_to_graph(relation, memory_item)
        metadata = {
                "memory_id": memory_item.id,
                "user_id": memory_item.user_id,
                "content": memory_item.content,
                "memory_type": memory_item.memory_type,
                "timestamp": int(memory_item.timestamp.timestamp()),
                "importance": memory_item.importance,
                "entities": [e.entity_id for e in entities],
                "entity_count": len(entities),
                "relation_count": len(relations)
            }
        status = self.vector_store.add_vectors(
            vectors=[embedding.tolist()],
            metadata=[metadata],
            ids=[memory_item.id]
        )
        if not status:
            print("⚠️\x20\x20记忆存储到Qdrant向量数据库失败，但已添加到Neo4j图数据库")
        memory_item.metadata["entities"] = [e.entity_id for e in entities]
        memory_item.metadata["relations"] = [
            f"{r.from_entity}-{r.relation_type}-{r.to_entity}" for r in relations
        ]
        self.semantic_memories.append(memory_item)
        print(f"✅\x20已成功添加语义记忆：{len(entities)}个实体, {len(relations)}条关系")
        return memory_item.id
    
    def retrieve(self, query: str, limit: int = 5, **kwargs) -> List[MemoryItem]:
        user_id = kwargs.get("user_id")
        if not user_id:
            print("⛔\x20输入的'user_id'为空")
            return []
        vector_results = self._vector_search(query, limit * 2, user_id)
        graph_results = self._graph_search(query, limit * 2, user_id)
        combined_results = self._combine_and_rank_results(vector_results, graph_results, query, limit)
        scores = [r.get("combined_score", r.get("vector_score", 0.0)) for r in combined_results]
        if scores:
            max_s = max(scores)
            exps = [math.exp(s - max_s) for s in scores]
            denom = sum(exps) or 1.0
            probs = [e / denom for e in exps]
        else:
            probs = []
        result_memories = []
        for idx, result in enumerate(combined_results):
            memory_id = result.get("memory_id")
            memory = next((m for m in self.semantic_memories if m.id == memory_id), None)
            if memory and memory.metadata.get("forgotten", False):
                continue
            timestamp = result.get("timestamp")
            memory_item = MemoryItem(
                id=result["memory_id"],
                content=result["content"],
                memory_type="semantic",
                user_id=result.get("user_id", "default"),
                timestamp=timestamp,
                importance=result.get("importance", 0.5),
                metadata={
                    **result.get("metadata", {}),
                    "combined_score": result.get("combined_score", 0.0),
                    "vector_score": result.get("vector_score", 0.0),
                    "graph_score": result.get("graph_score", 0.0),
                    "probability": probs[idx] if idx < len(probs) else 0.0,
                }
            )
            result_memories.append(memory_item)
        return result_memories[:limit]
    
    def _vector_search(self, query: str, limit: int, user_id: str) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_model.encode(query)
        where_filter = {"memory_type": "semantic", "user_id": user_id}
        results = self.vector_store.search_similar(
            query_vector=query_embedding.tolist(),
            limit=limit,
            where=where_filter
        )
        formatted_results = []
        for result in results:
            formatted_result = {
                "id": result["id"],
                "score": result["score"],
                **result["metadata"]
            }
            formatted_results.append(formatted_result)
        print(f"🔍\x20Qdrant向量数据库搜索返回{len(formatted_results)}个结果")
        return formatted_results

    def _graph_search(self, query: str, limit: int, user_id: Optional[str] = None) -> List[Dict[str, Any]]:
        query_entities = self._extract_entities(query)
        if not query_entities:
            entities_by_name = self.graph_store.search_entities_by_name(
                name_pattern=query, 
                limit=10
            )
            if entities_by_name:
                query_entities = [Entity(
                    entity_id=e["id"],
                    name=e["name"],
                    entity_type=e["type"]
                ) for e in entities_by_name[:3]]
            else:
                return []
        related_memory_ids = set()
        for entity in query_entities:
            related_entities = self.graph_store.find_related_entities(
                entity_id=entity.entity_id,
                max_depth=2,
                limit=20
            )
            for rel_entity in related_entities:
                if "memory_id" in rel_entity:
                    related_memory_ids.add(rel_entity["memory_id"])
            entity_rels = self.graph_store.get_entity_relationships(entity.entity_id)
            for rel in entity_rels:
                rel_data = rel.get("relationship", {})
                if "memory_id" in rel_data:
                    related_memory_ids.add(rel_data["memory_id"])
        results = []
        for memory_id in list(related_memory_ids)[:limit * 2]:
            mem = self._find_memory_by_id(memory_id)
            if not mem:
                continue
            if user_id and mem.user_id != user_id:
                continue
            metadata = {
                "content": mem.content,
                "user_id": mem.user_id,
                "memory_type": mem.memory_type,
                "importance": mem.importance,
                "timestamp": int(mem.timestamp.timestamp()),
                "entities": mem.metadata.get("entities", [])
            }
            graph_score = self._calculate_graph_relevance_neo4j(metadata, query_entities)
            results.append({
                "id": memory_id,
                "memory_id": memory_id,
                "content": metadata.get("content", ""),
                "similarity": graph_score,
                "user_id": metadata.get("user_id"),
                "memory_type": metadata.get("memory_type"),
                "importance": metadata.get("importance", 0.5),
                "timestamp": metadata.get("timestamp"),
                "entities": metadata.get("entities", [])
            })
        results.sort(key=lambda x: x["similarity"], reverse=True)
        ret_results = results[:limit]
        print(f"🔍\x20Neo4j图数据库搜索到{len(results)}个结果，返回{len(ret_results)}个结果")
        return ret_results

    def _combine_and_rank_results(
        self,
        vector_results: List[Dict[str, Any]],
        graph_results: List[Dict[str, Any]],
        query: str,
        limit: int
    ) -> List[Dict[str, Any]]:
        combined = {}
        content_seen = set()
        for result in vector_results:
            memory_id = result["memory_id"]
            content = result.get("content", "")
            content_hash = hash(content.strip())
            if content_hash in content_seen:
                continue
            content_seen.add(content_hash)
            combined[memory_id] = {
                **result,
                "vector_score": result.get("score", 0.0), 
                "graph_score": 0.0,
                "content_hash": content_hash
            }
        for result in graph_results:
            memory_id = result["memory_id"]
            content = result.get("content", "")
            content_hash = hash(content.strip())      
            if memory_id in combined:
                combined[memory_id]["graph_score"] = result.get("similarity", 0.0)
            elif content_hash not in content_seen:
                content_seen.add(content_hash)
                combined[memory_id] = {
                    **result,
                    "vector_score": 0.0,
                    "graph_score": result.get("similarity", 0.0),
                    "content_hash": content_hash
                }
        for memory_id, result in combined.items():
            vector_score = result["vector_score"]
            graph_score = result["graph_score"]
            importance = result.get("importance", 0.5)
            base_relevance = vector_score * 0.7 + graph_score * 0.3
            importance_weight = 0.8 + (importance * 0.4)
            combined_score = base_relevance * importance_weight
            result["debug_info"] = {
                "base_relevance": base_relevance,
                "importance_weight": importance_weight,
                "combined_score": combined_score
            }
            result["combined_score"] = combined_score
        min_threshold = 0.1
        filtered_results = [
            result for result in combined.values() 
            if result["combined_score"] >= min_threshold
        ]
        sorted_results = sorted(
            filtered_results,
            key=lambda x: x["combined_score"],
            reverse=True
        )
        return sorted_results[:limit]
    
    def _detect_language(self, text: str) -> str:
        chinese_chars = sum(1 for ch in text if '\u4e00' <= ch <= '\u9fff')
        total_chars = len(text.replace(' ', ''))
        if total_chars == 0:
            return "en"
        chinese_ratio = chinese_chars / total_chars
        return "zh" if chinese_ratio > 0.3 else "en"
    
    def _extract_entities(self, text: str) -> List[Entity]:
        entities = []
        lang = self._detect_language(text)
        selected_nlp = self.nlp_zh if lang == "zh" else self.nlp_en
        if selected_nlp:
            doc = selected_nlp(text)
            self._store_linguistic_analysis(doc, text)
            for ent in doc.ents:
                entity = Entity(
                    entity_id=f"entity_{hash(ent.text)}",
                    name=ent.text,
                    entity_type=ent.label_,
                    description=f"从文本中识别的{ent.label_}实体"
                )
                entities.append(entity)
        return entities
    
    def _store_linguistic_analysis(self, doc, text: str):
        if not self.graph_store:
            return
        for token in doc:
            if token.is_punct or token.is_space:
                continue 
            token_id = f"token_{hash(token.text + token.pos_)}"
            self.graph_store.add_entity(
                entity_id=token_id,
                name=token.text,
                entity_type="TOKEN",
                properties={
                    "pos": token.pos_,
                    "tag": token.tag_,
                    "lemma": token.lemma_,
                    "is_alpha": token.is_alpha,
                    "is_stop": token.is_stop,
                    "source_text": text[:50]
                }
            )
            if token.pos_ in ["NOUN", "PROPN"]:
                concept_id = f"concept_{hash(token.text)}"
                self.graph_store.add_entity(
                    entity_id=concept_id,
                    name=token.text,
                    entity_type="CONCEPT",
                    properties={
                        "category": token.pos_,
                        "frequency": 1,
                        "source_text": text[:50]
                    }
                )                  
                self.graph_store.add_relationship(
                    from_entity_id=token_id,
                    to_entity_id=concept_id,
                    relationship_type="REPRESENTS",
                    properties={"confidence": 1.0}
                )
        for token in doc:
            if token.is_punct or token.is_space or token.head == token:
                continue
            from_id = f"token_{hash(token.text + token.pos_)}"
            to_id = f"token_{hash(token.head.text + token.head.pos_)}"
            relation_type = token.dep_.upper().replace(":", "_")
            self.graph_store.add_relationship(
                from_entity_id=from_id,
                to_entity_id=to_id,
                relationship_type=relation_type,
                properties={
                    "dependency": token.dep_,
                    "source_text": text[:50]
                }
            )
    
    def _extract_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        relations = []
        for i, ent_start in enumerate(entities):
            for ent_end in entities[i+1:]:
                relations.append(Relation(
                    from_entity=ent_start.entity_id,
                    to_entity=ent_end.entity_id,
                    relation_type="CO_OCCURS",
                    strength=0.5,
                    evidence=text[:100]
                ))
        return relations
    
    def _add_entity_to_graph(self, entity: Entity, memory_item: MemoryItem):
        properties = {
            "name": entity.name,
            "description": entity.description,
            "frequency": entity.frequency,
            "memory_id": memory_item.id,
            "user_id": memory_item.user_id,
            "importance": memory_item.importance,
            **entity.properties
        }
        status = self.graph_store.add_entity(
            entity_id=entity.entity_id,
            name=entity.name,
            entity_type=entity.entity_type,
            properties=properties
        )
        if status:
            if entity.entity_id in self.entities:
                self.entities[entity.entity_id].frequency += 1
                self.entities[entity.entity_id].updated_at = datetime.now()
            else:
                self.entities[entity.entity_id] = entity
        return status
    
    def _add_relation_to_graph(self, relation: Relation, memory_item: MemoryItem):
        properties = {
            "strength": relation.strength,
            "memory_id": memory_item.id,
            "user_id": memory_item.user_id,
            "importance": memory_item.importance,
            "evidence": relation.evidence
        }
        status = self.graph_store.add_relationship(
            from_entity_id=relation.from_entity,
            to_entity_id=relation.to_entity,
            relationship_type=relation.relation_type,
            properties=properties
        )
        if status:
            self.relations.append(relation)
        return status
    
    def _calculate_graph_relevance_neo4j(self, memory_metadata: Dict[str, Any], query_entities: List[Entity]) -> float:
        memory_entities = memory_metadata.get("entities", [])
        if not memory_entities or not query_entities:
            return 0.0
        query_entity_ids = {e.entity_id for e in query_entities}
        matching_entities = len(set(memory_entities).intersection(query_entity_ids))
        entity_score = matching_entities / len(query_entity_ids) if query_entity_ids else 0
        entity_count = memory_metadata.get("entity_count", 0)
        entity_density = min(entity_count / 10, 1.0)
        relation_count = memory_metadata.get("relation_count", 0)
        relation_density = min(relation_count / 5, 1.0)
        relevance_score = entity_score * 0.6 + entity_density * 0.2 + relation_density * 0.2    
        return min(relevance_score, 1.0)

    def _add_or_update_entity(self, entity: Entity):
        if entity.entity_id in self.entities:
            existing = self.entities[entity.entity_id]
            existing.frequency += 1
            existing.updated_at = datetime.now()
        else:
            self.entities[entity.entity_id] = entity
    
    def _add_or_update_relation(self, relation: Relation):
        existing_relation = None
        for r in self.relations:
            if (r.from_entity == relation.from_entity and
                r.to_entity == relation.to_entity and
                r.relation_type == relation.relation_type):
                existing_relation = r
                break
        if existing_relation:
            existing_relation.frequency += 1
            existing_relation.strength = min(1.0, existing_relation.strength + 0.1)
        else:
            self.relations.append(relation)
    
    def _find_memory_by_id(self, memory_id: str) -> Optional[MemoryItem]:
        for memory in self.semantic_memories:
            if memory.id == memory_id:
                return memory
        print(f"⛔\x20未查询到记忆：ID={memory_id}, 记忆总数量={len(self.semantic_memories)}")
        return None
    
    def update(
        self,
        memory_id: str,
        content: str = None,
        importance: float = None,
        metadata: Dict[str, Any] = None
    ) -> bool:
        memory = self._find_memory_by_id(memory_id)
        if not memory:
            return False
        if content is not None:
            embedding = self.embedding_model.encode(content)
            self.memory_embeddings[memory_id] = embedding
            memory.content = content
            entities = self._extract_entities(content)
            relations = self._extract_relations(content, entities)
            for entity in entities:
                self._add_or_update_entity(entity)
            for relation in relations:
                self._add_or_update_relation(relation)
            memory.metadata["entities"] = [e.entity_id for e in entities]
            memory.metadata["relations"] = [
                f"{r.from_entity}-{r.relation_type}-{r.to_entity}" for r in relations
            ]
        if importance is not None:
            memory.importance = importance 
        if metadata is not None:
            memory.metadata.update(metadata)
        return True
    
    def remove(self, memory_id: str) -> bool:
        memory = self._find_memory_by_id(memory_id)
        if not memory:
            return False
        self.vector_store.delete_memories([memory_id])
        self.semantic_memories.remove(memory)
        del self.memory_embeddings[memory_id]
        return True
    
    def has_memory(self, memory_id: str) -> bool:
        return self._find_memory_by_id(memory_id) is not None
    
    def forget(self, strategy: str = "importance_based", threshold: float = 0.1, max_age_days: int = 30) -> int:
        forgotten_count = 0
        current_time = datetime.now()
        to_remove = []
        for memory in self.semantic_memories:
            should_forget = False
            if strategy == "importance_based":
                if memory.importance < threshold:
                    should_forget = True
            elif strategy == "time_based":
                cutoff_time = current_time - timedelta(days=max_age_days)
                if memory.timestamp < cutoff_time:
                    should_forget = True
            elif strategy == "capacity_based":
                if len(self.semantic_memories) > self.config.max_capacity:
                    sorted_memories = sorted(self.semantic_memories, key=lambda m: m.importance)
                    excess_count = len(self.semantic_memories) - self.config.max_capacity
                    if memory in sorted_memories[:excess_count]:
                        should_forget = True
            if should_forget:
                to_remove.append(memory.id)
        for memory_id in to_remove:
            if self.remove(memory_id):
                forgotten_count += 1
        return forgotten_count

    def clear(self):
        if self.vector_store:
            self.vector_store.clear_collection()
        if self.graph_store:
            self.graph_store.clear_all()
        self.semantic_memories.clear()
        self.memory_embeddings.clear()
        self.entities.clear()
        self.relations.clear() 
        print("🗑️\x20\x20语义记忆系统已完全清空")

    def get_all(self) -> List[MemoryItem]:
        return self.semantic_memories.copy()
    
    def get_stats(self) -> Dict[str, Any]:
        graph_stats = {}
        if self.graph_store:
            graph_stats = self.graph_store.get_stats()
        else:
            graph_stats = {}
        active_memories = self.semantic_memories
        return {
            "count": len(active_memories),
            "forgotten_count": 0,
            "total_count": len(self.semantic_memories),
            "entities_count": len(self.entities),
            "relations_count": len(self.relations),
            "graph_nodes": graph_stats.get("total_nodes", 0),
            "graph_edges": graph_stats.get("total_relationships", 0),
            "avg_importance": sum(m.importance for m in active_memories) / len(active_memories) if active_memories else 0.0,
            "memory_type": "enhanced_semantic"
        }
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        return self.entities.get(entity_id)
    
    def search_entities(self, query: str, limit: int = 10) -> List[Entity]:
        query_lower = query.lower()
        scored_entities = []
        for entity in self.entities.values():
            score = 0.0
            if query_lower in entity.name.lower():
                score += 2.0
            if query_lower in entity.entity_type.lower():
                score += 1.0
            if query_lower in entity.description.lower():
                score += 0.5
            score *= math.log(1 + entity.frequency)
            if score > 0:
                scored_entities.append((score, entity))
        scored_entities.sort(key=lambda x: x[0], reverse=True)
        return [entity for _, entity in scored_entities[:limit]]
    
    def get_related_entities(
        self,
        entity_id: str,
        relation_types: List[str] = None,
        max_hops: int = 2
    ) -> List[Dict[str, Any]]:
        related = []
        if not self.graph_store:
            return []
        related_entities = self.graph_store.find_related_entities(
            entity_id=entity_id,
            relationship_types=relation_types,
            max_depth=max_hops,
            limit=50
        )
        for entity_data in related_entities:
            entity_obj = self.entities.get(entity_data.get("id"))
            if not entity_obj:
                entity_obj = Entity(
                    entity_id=entity_data.get("id", entity_id),
                    name=entity_data.get("name", ""),
                    entity_type=entity_data.get("type", "MISC")
                )
                related.append({
                    "entity": entity_obj,
                    "relation_type": entity_data.get("relationship_path", ["RELATED"])[-1] if entity_data.get("relationship_path") else "RELATED",
                    "strength": 1.0 / max(entity_data.get("distance", 1), 1),
                    "distance": entity_data.get("distance", max_hops)
                })
        related.sort(key=lambda x: (x["distance"], -x["strength"]))
        return related
    
    def export_knowledge_graph(self) -> Dict[str, Any]:
        if self.graph_store:
            stats = self.graph_store.get_stats()
            return {
                "entities": {eid: entity.to_dict() for eid, entity in self.entities.items()},
                "relations": [relation.to_dict() for relation in self.relations],
                "graph_stats": {
                    "total_nodes": stats.get("total_nodes", 0),
                    "entity_nodes": stats.get("entity_nodes", 0),
                    "memory_nodes": stats.get("memory_nodes", 0),
                    "total_relationships": stats.get("total_relationships", 0),
                    "cached_entities": len(self.entities),
                    "cached_relations": len(self.relations)
                }
            }
        else:
            print(f"⛔\x20导出知识图谱失败")
            return {
                "entities": {},
                "relations": [],
                "graph_stats": {"error": str(e)}
            }