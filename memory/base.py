from abc import ABC, abstractmethod
from typing import List, Dict, Any
from datetime import datetime
from pydantic import BaseModel
from core import MemoryConfig, QdrantConfig, Neo4jConfig

class MemoryItem(BaseModel):
    id: str
    content: str
    memory_type: str
    user_id: str
    timestamp: datetime
    importance: float = 0.5
    metadata: Dict[str, Any] = {}

    class Config:
        arbitrary_types_allowed = True

class BaseMemory(ABC):
    def __init__(
            self, 
            config: MemoryConfig, 
            storage_backend=None
        ):
        self.config = config
        self.storage = storage_backend
        self.memory_type = self.__class__.__name__.lower().replace("memory", "")

    @abstractmethod
    def add(self, memory_item: MemoryItem) -> str:
        pass

    @abstractmethod
    def retrieve(
            self, 
            query: str, 
            limit: int = 5, 
            **kwargs
        ) -> List[MemoryItem]:
        pass

    @abstractmethod
    def update(
            self, 
            memory_id: str, 
            content: str = None,
            importance: float = None, 
            metadata: Dict[str, Any] = None
        ) -> bool:
        pass

    @abstractmethod
    def remove(self, memory_id: str) -> bool:
        pass

    @abstractmethod
    def has_memory(self, memory_id: str) -> bool:
        pass

    @abstractmethod
    def clear(self):
        pass

    @abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        pass

    def _generate_id(self) -> str:
        import uuid
        return str(uuid.uuid4())

    def _calculate_importance(
            self, 
            content: str, 
            base_importance: float = 0.5
        ) -> float:
        importance = base_importance
        if len(content) > 100:
            importance += 0.1
        important_keywords = ["重要", "关键", "必须", "注意", "警告", "错误"]
        if any(keyword in content for keyword in important_keywords):
            importance += 0.2
        return max(0.0, min(1.0, importance))

    def __str__(self) -> str:
        stats = self.get_stats()
        return f"{self.__class__.__name__}(count={stats.get('count', 0)})"

    def __repr__(self) -> str:
        return self.__str__()