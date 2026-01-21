from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from ..base import Neo4jConfig

try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import ServiceUnavailable, AuthError
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    GraphDatabase = None

class Neo4jGraphStore:
    def __init__(
        self,
        uri: Optional[str] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        database: Optional[str] = None
    ):
        if not NEO4J_AVAILABLE:
            raise ImportError("未安装neo4j>=5.0.0")
        config = Neo4jConfig.from_env()
        self.uri = uri or config.uri
        self.username = username or config.username
        self.password = password or config.password
        self.database = database or config.database
        self.driver = None
        self._initialize_driver(
            max_connection_lifetime=config.connect_lifetime,
            max_connection_pool_size=config.connect_pool_size,
            connection_acquisition_timeout=config.connect_timeout
        )
        self._create_indexes()
    
    def _initialize_driver(self, **config):
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.username, self.password), **config)
            self.driver.verify_connectivity()
            print(f"✅\x20已成功连接到Neo4j数据库：{self.uri}")
        except AuthError as e:
            print(f"⛔\x20Neo4j用户认证失败：{str(e)}")
            raise RuntimeError
        except ServiceUnavailable as e:
            print(f"⛔\x20Neo4j数据库服务不可用：{str(e)}")
            raise RuntimeError
        except Exception as e:
            print(f"⛔\x20Neo4j数据库连接失败：{str(e)}")
            raise RuntimeError
    
    def _create_indexes(self):
        indexes = [
            # 实体索引
            "CREATE INDEX entity_id_index IF NOT EXISTS FOR (e:Entity) ON (e.id)",
            "CREATE INDEX entity_name_index IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX entity_type_index IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            # 记忆索引
            "CREATE INDEX memory_id_index IF NOT EXISTS FOR (m:Memory) ON (m.id)",
            "CREATE INDEX memory_type_index IF NOT EXISTS FOR (m:Memory) ON (m.memory_type)",
            "CREATE INDEX memory_timestamp_index IF NOT EXISTS FOR (m:Memory) ON (m.timestamp)",
        ]
        with self.driver.session(database=self.database) as session:
            for index_query in indexes:
                try:
                    session.run(index_query)
                except Exception as e:
                    print(f"⚠️\x20\x20索引创建失败：{index_query}\n{str(e)}")
        print("✅\x20已完成Neo4j数据库索引创建")
    
    def add_entity(self, entity_id: str, name: str, entity_type: str, properties: Dict[str, Any] = None) -> bool:
        props = properties or {}
        props.update({
            "id": entity_id,
            "name": name,
            "type": entity_type,
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        }) 
        query = """
            MERGE (e:Entity {id: $entity_id})
            SET e += $properties
            RETURN e
            """
        with self.driver.session(database=self.database) as session:
            try:
                result = session.run(query, entity_id=entity_id, properties=props)
                record = result.single()
                if record:
                    return True
                else:
                    print(f"⛔\x20添加实体失败：返回结果为空")
                    return False
            except Exception as e:
                print(f"⛔\x20添加实体失败：{str(e)}")
                return False
    
    def add_relationship(
        self, 
        from_entity_id: str, 
        to_entity_id: str, 
        relationship_type: str,
        properties: Dict[str, Any] = None
    ) -> bool:
        props = properties or {}
        props.update({
            "type": relationship_type,
            "created_at": datetime.now(),
            "updated_at": datetime.now()
        })
        query = f"""
            MATCH (from:Entity {{id: $from_id}})
            MATCH (to:Entity {{id: $to_id}})
            MERGE (from)-[r:{relationship_type}]->(to)
            SET r += $properties
            RETURN r
            """
        with self.driver.session(database=self.database) as session:
            try:
                result = session.run(query, from_id=from_entity_id, to_id=to_entity_id, properties=props)
                record = result.single()
                if record:
                    return True
                else:
                    print(f"⛔\x20添加关系失败：返回结果为空")
                    return False
            except Exception as e:
                print(f"⛔\x20添加关系失败：{str(e)}")
                return False
    
    def find_related_entities(
        self, 
        entity_id: str, 
        relationship_types: List[str] = None,
        max_depth: int = 2,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        rel_filter = ""
        if relationship_types:
            rel_types = "|".join(relationship_types)
            rel_filter = f":{rel_types}"
        query = f"""
            MATCH path = (start:Entity {{id: $entity_id}})-[r{rel_filter}*1..{max_depth}]-(related:Entity)
            WHERE start.id <> related.id
            RETURN DISTINCT related, 
                   length(path) as distance,
                   [rel in relationships(path) | type(rel)] as relationship_path
            ORDER BY distance, related.name
            LIMIT $limit
            """
        with self.driver.session(database=self.database) as session:
            try:
                result = session.run(query, entity_id=entity_id, limit=limit)
            except Exception as e:
                print(f"⛔\x20查找关联实体失败：{str(e)}")
                return []
            entities = []
            for record in result:
                entity_data = dict(record["related"])
                entity_data["distance"] = record["distance"]
                entity_data["relationship_path"] = record["relationship_path"]
                entities.append(entity_data)
            return entities
    
    def search_entities_by_name(self, name_pattern: str, entity_types: List[str] = None, limit: int = 20) -> List[Dict[str, Any]]:
        type_filter = ""
        params = {"pattern": f".*{name_pattern}.*", "limit": limit}
        if entity_types:
            type_filter = "AND e.type IN $types"
            params["types"] = entity_types
        query = f"""
            MATCH (e:Entity)
            WHERE e.name =~ $pattern {type_filter}
            RETURN e
            ORDER BY e.name
            LIMIT $limit
            """
        with self.driver.session(database=self.database) as session:
            try:
                result = session.run(query, **params)
            except Exception as e:
                print(f"⛔\x20按名称搜索实体失败：{str(e)}")
                return []
            entities = []
            for record in result:
                entity_data = dict(record['e'])
                entities.append(entity_data)               
            return entities
    
    def get_entity_relationships(self, entity_id: str) -> List[Dict[str, Any]]:
        query = """
            MATCH (e:Entity {id: $entity_id})-[r]-(other:Entity)
            RETURN r, other, 
                   CASE WHEN startNode(r).id = $entity_id THEN 'outgoing' ELSE 'incoming' END as direction
            """
        with self.driver.session(database=self.database) as session:
            try:
                result = session.run(query, entity_id=entity_id)
            except Exception as e:
                print(f"⛔\x20获取实体关系失败：{str(e)}")
                return []                
            relationships = []
            for record in result:
                rel_data = dict(record['r'])
                other_data = dict(record["other"])
                rel = {
                    "relationship": rel_data,
                    "other_entity": other_data,
                    "direction": record["direction"]
                }
                relationships.append(rel) 
            return relationships
    
    def delete_entity(self, entity_id: str) -> bool:
        query = """
            MATCH (e:Entity {id: $entity_id})
            DETACH DELETE e
            """
        with self.driver.session(database=self.database) as session:
            try:
                result = session.run(query, entity_id=entity_id)
            except Exception as e:
                print(f"⛔\x20删除实体失败：{str(e)}")
                return False            
            summary = result.consume()
            deleted_count = summary.counters.nodes_deleted
            print(f"✅\x20已成功删除实体：{entity_id}（同时删除{deleted_count}个节点）")
            return deleted_count > 0
    
    def clear_all(self) -> bool:
        query = "MATCH (n) DETACH DELETE n"
        with self.driver.session(database=self.database) as session:
            try:
                result = session.run(query)
            except Exception as e:
                print(f"⛔\x20清空Neo4j数据库失败：{str(e)}")
                return False
            summary = result.consume()
            deleted_nodes = summary.counters.nodes_deleted
            deleted_relationships = summary.counters.relationships_deleted
            print(f"✅\x20已清空Neo4j数据库：删除{deleted_nodes}个节点，删除{deleted_relationships}条关系")
            return True
    
    def get_stats(self) -> Dict[str, Any]:
        queries = {
            "total_nodes": "MATCH (n) RETURN count(n) as count",
            "total_relationships": "MATCH ()-[r]->() RETURN count(r) as count",
            "entity_nodes": "MATCH (n:Entity) RETURN count(n) as count",
            "memory_nodes": "MATCH (n:Memory) RETURN count(n) as count",
        } 
        stats = {}
        with self.driver.session(database=self.database) as session:
            try:
                for key, query in queries.items():
                    result = session.run(query)
                    record = result.single()
                    stats[key] = record["count"] if record else 0
            except Exception as e:
                print(f"⛔\x20获取Neo4j数据库统计信息失败：{str(e)}")
                return {}
            return stats
    
    def heartbeat_check(self) -> bool:
        with self.driver.session(database=self.database) as session:
            try:
                result = session.run("RETURN 1 as health")
                record = result.single()
                return record["health"] == 1
            except Exception as e:
                print(f"⛔\x20Neo4j数据库心跳检测失败：{str(e)}")
            return False
 
    def __del__(self):
        if hasattr(self, "driver") and self.driver:
            try:
                self.driver.close()
            except:
                pass