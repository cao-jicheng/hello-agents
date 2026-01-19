import sys
sys.path.append("..")
from memory.storage import Neo4jGraphStore, QdrantConnectionManager

print('-'*60)
neo4j = Neo4jGraphStore()
print(f"💓\x20Neo4j图数据库通过心跳检测：{neo4j.heartbeat_check()}")

print('-'*60)
qdrant = QdrantConnectionManager.get_instance()
print(f"💓\x20Qdrant向量数据库通过心跳检测：{qdrant.heartbeat_check()}")