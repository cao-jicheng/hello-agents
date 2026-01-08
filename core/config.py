import os
from typing import Optional, Dict, Any
from pydantic import BaseModel

class Config(BaseModel):
    debug: bool = False
    log_level: str = "INFO"
    max_history_length: int = 100
    
    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            debug=os.getenv("DEBUG", "false").lower() == "true",
            log_level=os.getenv("LOG_LEVEL", "INFO"),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        return self.dict()
