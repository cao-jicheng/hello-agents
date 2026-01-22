import sys
sys.path.append("..")
from memory import get_text_embedder
from memory import SQLiteDocumentStore
from memory import Neo4jGraphStore, QdrantConnectionManager

print('-'*60)
embedding = get_text_embedder()
vec = embedding.encode("heartbeat_check")
print(f"💓\x20Embedding模型通过心跳检测：name={embedding.model}, dim={len(vec)}")

print('-'*60)
sqlite = SQLiteDocumentStore()
print(f"💓\x20SQLite数据库通过心跳检测：{hasattr(sqlite, '_initialized')}")
sqlite.close()

print('-'*60)
neo4j = Neo4jGraphStore()
print(f"💓\x20Neo4j图数据库通过心跳检测：{neo4j.heartbeat_check()}")
neo4j.clear_all()

print('-'*60)
qdrant = QdrantConnectionManager.get_instance()
print(f"💓\x20Qdrant向量数据库通过心跳检测：{qdrant.heartbeat_check()}")
qdrant.clear_collection()