import sys
sys.path.append("..")
import os
import threading
import requests
import numpy as np
from typing import List, Union, Optional
from core import EmbeddingConfig

class EmbeddingModel:
    def encode(self, texts: Union[str, List[str]]):
        raise NotImplementedError

    @property
    def dimension(self) -> int:
        raise NotImplementedError

class TransformerEmbedding(EmbeddingModel):
    def __init__(self, model_name: str = None):
        self.model_name = model_name or "sentence-transformers/all-MiniLM-L6-v2"
        self._backend = None
        self._st_model = None
        self._hf_tokenizer = None
        self._hf_model = None
        self._dimension = None
        self._load_backend()

    def _load_backend(self):
        try:
            from sentence_transformers import SentenceTransformer
            self._st_model = SentenceTransformer(self.model_name)
            test_vec = self._st_model.encode("test_text")
            self._dimension = len(test_vec)
            self._backend = "st"
            return
        except Exception:
            self._st_model = None

        try:
            import torch
            from transformers import AutoTokenizer, AutoModel 
            self._hf_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._hf_model = AutoModel.from_pretrained(self.model_name)
            with torch.no_grad():
                inputs = self._hf_tokenizer("test_text", return_tensors="pt", padding=True, truncation=True)
                outputs = self._hf_model(**inputs)
                test_embedding = outputs.last_hidden_state.mean(dim=1)
                self._dimension = int(test_embedding.shape[1])
            self._backend = "hf"
            return
        except Exception:
            self._hf_tokenizer = None
            self._hf_model = None
        raise ImportError("未找到可用的Transformer嵌入后端，请安装 sentence-transformers 或 transformers + torch")

    def encode(self, texts: Union[str, List[str]]):
        if isinstance(texts, str):
            inputs = [texts]
            single = True
        else:
            inputs = list(texts)
            single = False

        if self._backend == "st":
            vecs = self._st_model.encode(inputs)
            if hasattr(vecs, "tolist"):
                vecs = [v for v in vecs]
        else:
            import torch
            tokenized = self._hf_tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self._hf_model(**tokenized)
                embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
            vecs = [v for v in embeddings]
        if single:
            return vecs[0]
        return vecs

    @property
    def dimension(self) -> int:
        return int(self._dimension or 0)

class TFIDFEmbedding(EmbeddingModel):
    def __init__(self, max_features: int = 1000):
        self.max_features = max_features
        self._vectorizer = None
        self._is_fitted = False
        self._dimension = max_features
        self._init_vectorizer()

    def _init_vectorizer(self):
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            self._vectorizer = TfidfVectorizer(max_features=self.max_features, stop_words="english")
        except Exception:
            raise ImportError("未找到TFIDF依赖库，请安装 scikit-learn")

    def fit(self, texts: List[str]):
        self._vectorizer.fit(texts)
        self._is_fitted = True
        self._dimension = len(self._vectorizer.get_feature_names_out())

    def encode(self, texts: Union[str, List[str]]):
        if not self._is_fitted:
            raise ValueError("TFIDF模型未训练，请先调用fit()方法")
        if isinstance(texts, str):
            texts = [texts]
            single = True
        else:
            single = False
        tfidf_matrix = self._vectorizer.transform(texts)
        embeddings = tfidf_matrix.toarray()
        if single:
            return embeddings[0]
        return [e for e in embeddings]

    @property
    def dimension(self) -> int:
        return int(self._dimension or 0)

class RestAPIEmbedding(EmbeddingModel):
    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None        
        ):
        config = EmbeddingConfig.from_env()
        self.model = model or config.model
        self.base_url = base_url or config.base_url
        self.api_key = api_key or config.api_key
        if not all([self.model, self.base_url, self.api_key]):
            raise Exception("Embedding模型名称、访问网址、API密钥需要显式指定或在.env文件中定义")
        test = self.encode("模型维度嗅探")
        self._dimension = len(test)

    def encode(self, texts: Union[str, List[str]]):
        if isinstance(texts, str):
            inputs = [texts]
            single = True
        else:
            inputs = list(texts)
            single = False
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "input": inputs
        }
        try:
            response = requests.post(self.base_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
        except Exception as e:
            print(f"⛔\x20Embedding API调用失败：{str(e)}")
            return []
        data = response.json()
        items = data.get("data") or []
        vecs = [np.array(item.get("embedding")) for item in items]
        return vecs[0] if single else vecs

    @property
    def dimension(self) -> int:
        return int(self._dimension or 0)

def create_embedding_model(model_type: str, **kwargs) -> EmbeddingModel:
    if model_type == "restapi":
        return RestAPIEmbedding()
    elif model_type == "local":
        return TransformerEmbedding(**kwargs)
    elif model_type == "tfidf":
        return TFIDFEmbedding(**kwargs)
    else:
        print(f"⚠️\x20\x20不支持{model_type}类型，使用restapi代替")
        return RestAPIEmbedding()

def _build_embedder(**kwargs) -> EmbeddingModel:
    priority = ["restapi", "local", "tfidf"]
    for t in priority:
        try:
            return create_embedding_model(t, **kwargs)
        except Exception:
            continue
    raise RuntimeError("所有Embedding模型都不可用，请安装依赖或检查配置")

_lock = threading.RLock()
_embedder: Optional[EmbeddingModel] = None

def get_text_embedder() -> EmbeddingModel:
    global _embedder
    if _embedder is not None:
        return _embedder
    with _lock:
        if _embedder is None:
            _embedder = _build_embedder()
        return _embedder

def get_dimension(default: int = 384) -> int:
    try:
        return int(getattr(get_text_embedder(), "dimension", default))
    except Exception:
        return int(default)

def refresh_embedder() -> EmbeddingModel:
    global _embedder
    with _lock:
        _embedder = _build_embedder()
        return _embedder