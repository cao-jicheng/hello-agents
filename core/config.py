import os
from dotenv import load_dotenv
from typing import Dict, List, Any
from pydantic import BaseModel, Field

load_dotenv()

class LLMConfig(BaseModel):
    model: str = Field(
        description="LLM模型名称"
    )
    base_url: str = Field(
        description="LLM API访问地址"
    )
    api_key: str = Field(
        description="LLM API访问密钥"
    )
    timeout: int = Field(
        description="LLM访问超时（秒）"
    )

    @classmethod
    def from_env(cls) -> "LLMConfig":
        return cls(
            model=os.getenv("LLM_MODEL", "deepseek-chat"),
            base_url=os.getenv("LLM_BASE_URL", "https://api.deepseek.com/v1"),
            api_key=os.getenv("LLM_API_KEY"),
            timeout=int(os.getenv("LLM_TIMEOUT", "60"))
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True)

class EmbeddingConfig(BaseModel):
    model: str = Field(
        description="Embedding模型名称"
    )
    base_url: str = Field(
        description="Embedding模型API访问地址"
    )
    api_key: str = Field(
        description="Embedding模型API访问密钥"
    )

    @classmethod
    def from_env(cls) -> "EmbeddingConfig":
        return cls(
            model=os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
            base_url=os.getenv("EMBEDDING_BASE_URL", "https://api.siliconflow.cn/v1/embeddings"),
            api_key=os.getenv("EMBEDDING_API_KEY")
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True)

class SearchConfig(BaseModel):
    tavily_api_key: str = Field(
        description="Tavily搜索引擎API访问密钥"
    )
    bocha_api_key: str = Field(
        description="博查搜索引擎API访问密钥"
    )

    @classmethod
    def from_env(cls) -> "SearchConfig":
        return cls(
            tavily_api_key=os.getenv("TAVILY_API_KEY"),
            bocha_api_key=os.getenv("BOCHA_API_KEY")
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True)

class MemoryConfig(BaseModel):
    storage_path: str = "./memory_data"
    max_capacity: int = 100
    importance_threshold: float = 0.1
    decay_factor: float = 0.95
    working_memory_capacity: int = 10
    working_memory_tokens: int = 2000
    working_memory_ttl_minutes: int = 120
    perceptual_memory_modalities: List[str] = ["text", "image", "audio", "video"]

class QdrantConfig(BaseModel):
    url: str = Field(
        description="Qdrant服务URL"
    )
    collection_name: str = Field(
        description="向量集合名称"
    )
    vector_size: int = Field(
        description="向量维度"
    )
    distance: str = Field(
        description="距离度量方式 (cosine, dot, euclidean)"
    )
    timeout: int = Field(
        description="连接超时（秒）"
    )
    hnsw_m: int = Field(
        description="HNSW索引（每个节点的最大连接数）"
    )
    hnsw_ef_construct: int = Field(
        description="HNSW索引（索引构建时的候选邻居数量）"
    )
    hnsw_ef_search: int = Field(
        description="HNSW索引（索引搜索时的候选邻居数量）"
    )
    exact_search: str = Field(
        description="精准搜索"
    )
    
    @classmethod
    def from_env(cls) -> "QdrantConfig":
        return cls(
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
            collection_name=os.getenv("QDRANT_COLLECTION", "hello_agents_vectors"),
            vector_size=int(os.getenv("QDRANT_VECTOR_SIZE", "384")),
            distance=os.getenv("QDRANT_DISTANCE", "cosine"),
            timeout=int(os.getenv("QDRANT_TIMEOUT", "30")),
            hnsw_m=int(os.getenv("QDRANT_HNSW_M", "32")),
            hnsw_ef_construct=int(os.getenv("QDRANT_HNSW_EF_CONSTRUCT", "256")),
            hnsw_ef_search=int(os.getenv("QDRANT_HNSW_EF_SEARCH", "128")),
            exact_search=os.getenv("QDRANT_EXACT_SEARCH", "0")
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True)


class Neo4jConfig(BaseModel):
    uri: str = Field(
        description="Neo4j连接URI"
    )
    username: str = Field(
        description="用户名"
    )
    password: str = Field(
        description="登录密码"
    )
    database: str = Field(
        description="数据库名称"
    )
    connect_lifetime: int = Field(
        description="最大连接生命周期（秒）"
    )
    connect_pool_size: int = Field(
        description="最大连接池大小"
    )
    connect_timeout: int = Field(
        description="连接超时（秒）"
    )
    
    @classmethod
    def from_env(cls) -> "Neo4jConfig":
        return cls(
            uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
            username=os.getenv("NEO4J_USERNAME"),
            password=os.getenv("NEO4J_PASSWORD"),
            database=os.getenv("NEO4J_DATABASE", "neo4j"),
            connect_lifetime=int(os.getenv("NEO4J_CONNECT_LIFETIME", "3600")),
            connect_pool_size=int(os.getenv("NEO4J_CONNECT_POOL_SIZE", "50")),
            connect_timeout=int(os.getenv("NEO4J_CONNECT_TIMEOUT", "60"))
        )

    def to_dict(self) -> Dict[str, Any]:
        return self.model_dump(exclude_none=True)


class DatabaseConfig(BaseModel):
    qdrant: QdrantConfig = Field(
        description="Qdrant向量数据库配置"
    )
    neo4j: Neo4jConfig = Field(
        description="Neo4j图数据库配置"
    )
    
    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        return cls(
            qdrant=QdrantConfig.from_env(),
            neo4j=Neo4jConfig.from_env()
        )

    def get_qdrant_config(self) -> Dict[str, Any]:
        return self.qdrant.to_dict()
    
    def get_neo4j_config(self) -> Dict[str, Any]:
        return self.neo4j.to_dict()