import os
import uuid
import threading
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from ..base import QdrantConfig

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.models import (
        Distance, VectorParams, PointStruct, 
        Filter, FieldCondition, MatchValue, SearchRequest
    )
    QDRANT_AVAILABLE = True
except ImportError:
    QDRANT_AVAILABLE = False
    QdrantClient = None
    models = None

class QdrantConnectionManager:
    _instances = {}
    _lock = threading.Lock()
    
    @classmethod
    def get_instance(
        cls,
        url: Optional[str] = None,
        collection_name: str = "hello_agents_vectors"
    ) -> "QdrantVectorStore":
        key = (url or "local", collection_name)
        if key not in cls._instances:
            with cls._lock:
                if key not in cls._instances:
                    print(f"✏️\x20\x20创建新的Qdrant连接：{collection_name}")
                    cls._instances[key] = QdrantVectorStore()
                else:
                    print(f"♻️\x20\x20复用现有Qdrant连接：{collection_name}")
        else:
            print(f"♻️\x20\x20复用现有Qdrant连接：{collection_name}")    
        return cls._instances[key]

class QdrantVectorStore:
    def __init__(
        self,
        url: Optional[str] = None,
        collection_name: Optional[str] = None,
        vector_size: Optional[int] = None
        ):
        if not QDRANT_AVAILABLE:
            raise ImportError("未安装qdrant-client>=1.6.0")
        config = QdrantConfig.from_env()
        self.url = url or config.url
        self.collection_name = collection_name or config.collection_name
        self.vector_size = vector_size or config.vector_size
        self.timeout = config.timeout
        self.hnsw_m = config.hnsw_m
        self.hnsw_ef_construct = config.hnsw_ef_construct
        self.hnsw_ef_search = config.hnsw_ef_search
        self.exact_search = (config.exact_search == "1")
        distance_map = {
            "cosine": Distance.COSINE,
            "dot": Distance.DOT,
            "euclidean": Distance.EUCLID,
        }
        self.distance = distance_map.get(config.distance.lower(), Distance.COSINE)
        self.client = None
        self._initialize_client()
        
    def _initialize_client(self):
        self.client = QdrantClient(url=self.url, timeout=self.timeout)
        print(f"✅\x20已成功连接到Qdrant服务：{self.url}")
        self._ensure_collection()
    
    def _ensure_collection(self):
        collections = self.client.get_collections().collections
        collection_names = [c.name for c in collections]
        if self.collection_name not in collection_names:
            hnsw_cfg = None
            try:
                hnsw_cfg = models.HnswConfigDiff(m=self.hnsw_m, ef_construct=self.hnsw_ef_construct)
            except Exception:
                pass
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=self.distance
                ),
                hnsw_config=hnsw_cfg
            )
            print(f"✏️\x20\x20创建新的Qdrant集合：{self.collection_name}")
        else:
            print(f"♻️\x20\x20复用现有Qdrant集合：{self.collection_name}")
            try:
                self.client.update_collection(
                    collection_name=self.collection_name,
                    hnsw_config=models.HnswConfigDiff(m=self.hnsw_m, ef_construct=self.hnsw_ef_construct)
                )
            except Exception as e:
                print(f"⚠️\x20\x20更新HNSW配置失败：{e}")
        self._ensure_payload_indexes()
        print(f"✅\x20已成功初始化Qdrant集合")

    def _ensure_payload_indexes(self):
        index_fields = [
            ("memory_type", models.PayloadSchemaType.KEYWORD),
            ("user_id", models.PayloadSchemaType.KEYWORD),
            ("memory_id", models.PayloadSchemaType.KEYWORD),
            ("timestamp", models.PayloadSchemaType.INTEGER),
            ("modality", models.PayloadSchemaType.KEYWORD),
            ("source", models.PayloadSchemaType.KEYWORD),
            ("external", models.PayloadSchemaType.BOOL),
            ("namespace", models.PayloadSchemaType.KEYWORD),
            ("is_rag_data", models.PayloadSchemaType.BOOL),
            ("rag_namespace", models.PayloadSchemaType.KEYWORD),
            ("data_source", models.PayloadSchemaType.KEYWORD),
        ]
        for field_name, schema_type in index_fields:
            try:
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name=field_name,
                    field_schema=schema_type
                )
            except Exception as e:
                print(f"⚠️\x20\x20索引创建失败：{field_name}\n{str(e)}")
    
    def add_vectors(
        self, 
        vectors: List[List[float]], 
        metadata: List[Dict[str, Any]], 
        ids: Optional[List[int]] = None
    ) -> bool:
        if not vectors:
            print("⛔\x20输入的向量列表为空")
            return False
        if ids is None:
            ids = ["fake_id" for _ in range(len(vectors))]
        print(f"[Qdrant] add_vectors：n_vectors={len(vectors)} n_meta={len(metadata)} collection={self.collection_name}")
        points = []
        for i, (vector, meta, point_id) in enumerate(zip(vectors, metadata, ids)):
            if not isinstance(vector, list):
                print(f"⚠️\x20\x20非法向量类型：index={i} type={type(vector)} value={vector}")
                continue
            if len(vector) != self.vector_size:
                print(f"⚠️\x20\x20向量维度不匹配：期望{self.vector_size} 实际{len(vector)}")
                continue
            timestamp = int(datetime.now().timestamp())
            meta_with_timestamp = meta.copy()
            meta_with_timestamp["timestamp"] = timestamp
            meta_with_timestamp["added_at"] = timestamp
            if "external" in meta_with_timestamp and not isinstance(meta_with_timestamp.get("external"), bool):
                val = meta_with_timestamp.get("external")
                meta_with_timestamp["external"] = True if str(val).lower() in ("1", "true", "yes") else False
            safe_id: Any
            if isinstance(point_id, int):
                safe_id = point_id
            else:
                safe_id = str(uuid.uuid4())
            point = PointStruct(
                id=safe_id,
                vector=vector,
                payload=meta_with_timestamp
            )
            points.append(point)
        if not points:
            print("⛔\x20没有生成有效的向量点，操作终止")
            return False
        print(f"[Qdrant] upsert：points={len(points)}")
        operation_info = self.client.upsert(
            collection_name=self.collection_name,
            points=points,
            wait=True
        )
        print(f"✅\x20已成功添加{len(points)}个向量到Qdrant数据库")
        return True
    
    def search_similar(
        self, 
        query_vector: List[float], 
        limit: int = 10, 
        score_threshold: Optional[float] = None,
        where: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        if len(query_vector) != self.vector_size:
            print(f"⛔\x20查询向量维度错误：期望{self.vector_size} 实际{len(query_vector)}")
            return []
        query_filter = None
        if where:
            conditions = []
            for key, value in where.items():
                if isinstance(value, (str, int, float, bool)):
                    conditions.append(
                        FieldCondition(
                            key=key,
                            match=MatchValue(value=value)
                        )
                    )
            if conditions:
                query_filter = Filter(must=conditions)
        search_params = None
        try:
            search_params = models.SearchParams(hnsw_ef=self.hnsw_ef_search, exact=self.exact_search)
        except Exception:
            search_params = None
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            query_filter=query_filter,
            limit=limit,
            score_threshold=score_threshold,
            with_payload=True,
            with_vectors=False,
            search_params=search_params
        )
        search_result = response.points
        results = []
        for hit in search_result:
            result = {
                "id": hit.id,
                "score": hit.score,
                "metadata": hit.payload or {}
            }
            results.append(result)
        print(f"🔍\x20Qdrant数据库返回{len(results)}个搜索结果")
        return results
    
    def delete_vectors(self, ids: List[str]) -> bool:
        if not ids:
            return True
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=ids),
                wait=True
            )
            print(f"✅\x20已成功删除{len(ids)}个向量")
            return True
        except Exception as e:
            print(f"⛔\x20删除向量失败：{str(e)}")
            return False
    
    def clear_collection(self) -> bool:
        try:
            self.client.delete_collection(collection_name=self.collection_name)
            print(f"✅\x20已成功删除Qdrant集合：{self.collection_name}")
            self._ensure_collection()
            return True
        except Exception as e:
            print(f"⛔\x20清空集合失败：{str(e)}")
            return False
    
    def delete_memories(self, memory_ids: List[str]) -> bool:
        if not memory_ids:
            return True
        conditions = [FieldCondition(key="memory_id", match=MatchValue(value=mid)) for mid in memory_ids]
        query_filter = Filter(should=conditions)
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.FilterSelector(filter=query_filter),
                wait=True
            )
            print(f"✅\x20已成功删除{len(memory_ids)}个记忆")
            return True
        except Exception as e:
            print(f"⛔\x20删除记忆失败：{str(e)}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        try:
            collection_info = self.client.get_collection(self.collection_name)     
            return {
                "name": self.collection_name,
                "vectors_count": collection_info.vectors_count,
                "indexed_vectors_count": collection_info.indexed_vectors_count,
                "points_count": collection_info.points_count,
                "segments_count": collection_info.segments_count,
                "config": {
                    "vector_size": self.vector_size,
                    "distance": self.distance.value,
                }
            }     
        except Exception as e:
            print(f"⛔\x20获取集合信息失败：{str(e)}")
            return {}
    
    def get_collection_stats(self) -> Dict[str, Any]:
        info = self.get_collection_info()
        if info:
            info["store_type"] = "qdrant"
            return info
        else:
            return {"store_type": "qdrant", "name": self.collection_name}
    
    def heartbeat_check(self) -> bool:
        try:
            self.client.get_collections()
            return True
        except Exception as e:
            print(f"⛔\x20Qdrant数据库心跳检测失败：{str(e)}")
            return False
    
    def __del__(self):
        if hasattr(self, "client") and self.client:
            try:
                self.client.close()
            except:
                pass