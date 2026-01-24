import os
import time
from typing import Dict, Any, List, Optional
from core import Tool, ToolParameter, tool_action, OpenAICompatibleLLM
from memory import create_rag_pipeline

class RAGTool(Tool):
    def __init__(
        self,
        knowledge_base_path: str = "./knowledge_base",
        collection_name: str = "rag_knowledge_base",
        rag_namespace: str = "default",
        expandable: bool = False
    ):
        super().__init__(
            name="rag",
            description="RAG工具，支持多格式文档检索增强生成，提供智能问答能力",
            expandable=expandable
        )
        self.knowledge_base_path = os.path.abspath(knowledge_base_path)
        self.collection_name = collection_name
        self.rag_namespace = rag_namespace
        self._pipelines: Dict[str, Dict[str, Any]] = {}
        os.makedirs(knowledge_base_path, exist_ok=True)
        self._init_components()
    
    def _init_components(self):
        try:
            default_pipeline = create_rag_pipeline(
                collection_name=self.collection_name,
                rag_namespace=self.rag_namespace
            )
            self._pipelines[self.rag_namespace] = default_pipeline
            self.llm = OpenAICompatibleLLM()
            self.initialized = True
            print(f"[RAGTool] 已成功初始化：namespace={self.rag_namespace}, collection={self.collection_name}")      
        except Exception as e:
            self.initialized = False
            print(f"[RAGTool] ⛔\x20初始化失败：{str(e)}")

    def _get_pipeline(self, namespace: Optional[str] = None) -> Dict[str, Any]:
        target_ns = namespace or self.rag_namespace
        if target_ns in self._pipelines:
            return self._pipelines[target_ns]
        pipeline = create_rag_pipeline(
            collection_name=self.collection_name,
            rag_namespace=target_ns
        )
        self._pipelines[target_ns] = pipeline
        return pipeline

    def run(self, parameters: Dict[str, Any]) -> str:
        if not self.validate_parameters(parameters):
            return "参数验证失败：缺少必需的参数"
        if not self.initialized:
            return f"RAG工具未正确初始化，请检查配置"
        action = parameters.get("action")
        if action == "add_document":
            return self._add_document(
                file_path=parameters.get("file_path"),
                document_id=parameters.get("document_id"),
                namespace=parameters.get("namespace", "default"),
                chunk_size=parameters.get("chunk_size", 800),
                chunk_overlap=parameters.get("chunk_overlap", 100)
            )
        elif action == "add_text":
            return self._add_text(
                text=parameters.get("text"),
                document_id=parameters.get("document_id"),
                namespace=parameters.get("namespace", "default"),
                chunk_size=parameters.get("chunk_size", 800),
                chunk_overlap=parameters.get("chunk_overlap", 100)
            )
        elif action == "ask":
            question = parameters.get("question") or parameters.get("query")
            return self._ask(
                question=question,
                limit=parameters.get("limit", 5),
                enable_advanced_search=parameters.get("enable_advanced_search", True),
                include_citations=parameters.get("include_citations", True),
                max_chars=parameters.get("max_chars", 1200),
                namespace=parameters.get("namespace", "default")
            )
        elif action == "search":
            return self._search(
                query=parameters.get("query") or parameters.get("question"),
                limit=parameters.get("limit", 5),
                min_score=parameters.get("min_score", 0.1),
                enable_advanced_search=parameters.get("enable_advanced_search", True),
                max_chars=parameters.get("max_chars", 1200),
                include_citations=parameters.get("include_citations", True),
                namespace=parameters.get("namespace", "default")
            )
        elif action == "stats":
            return self._get_stats(namespace=parameters.get("namespace", "default"))
        elif action == "clear":
            return self._clear_knowledge_base(
                confirm=parameters.get("confirm", False),
                namespace=parameters.get("namespace", "default")
            )
        else:
            return f"运行出错：不支持的操作{action}"

    def get_parameters(self) -> List[ToolParameter]:
        return [
            ToolParameter(
                name="action",
                type="string",
                description="操作类型：add_document(添加文档), add_text(添加文本), ask(智能问答), search(搜索), stats(统计), clear(清空)",
                required=True
            ),
            ToolParameter(
                name="file_path",
                type="string",
                description="文档文件路径（支持PDF、Word、Excel、PPT、图片、音频等多种格式）",
                required=False
            ),
            ToolParameter(
                name="text",
                type="string",
                description="要添加的文本内容",
                required=False
            ),
            ToolParameter(
                name="question",
                type="string", 
                description="用户问题（用于智能问答）",
                required=False
            ),
            ToolParameter(
                name="query",
                type="string",
                description="搜索查询词（用于基础搜索）",
                required=False
            ),
            ToolParameter(
                name="namespace",
                type="string",
                description="知识库命名空间（用于隔离不同项目，默认：default）",
                required=False,
                default="default"
            ),
            ToolParameter(
                name="limit",
                type="integer",
                description="返回结果数量（默认：5）",
                required=False,
                default=5
            ),
            ToolParameter(
                name="include_citations",
                type="boolean",
                description="是否包含引用来源（默认：true）",
                required=False,
                default=True
            )
        ]

    @tool_action("rag_add_document", "添加文档到知识库（支持PDF、Word、Excel、PPT、图片、音频等多种格式）")
    def _add_document(
        self,
        file_path: str,
        document_id: str = None,
        namespace: str = "default",
        chunk_size: int = 800,
        chunk_overlap: int = 100
    ) -> str:
        """添加文档到知识库

        Args:
            file_path: 文档文件路径
            document_id: 文档ID（可选）
            namespace: 知识库命名空间（用于隔离不同项目）
            chunk_size: 分块大小
            chunk_overlap: 分块重叠大小

        Returns:
            执行结果
        """
        if not file_path or not os.path.exists(file_path):
            return f"文件不存在：{file_path}"
        pipeline = self._get_pipeline(namespace)
        t0 = time.time()
        chunks_added = pipeline["add_documents"](
            file_paths=[file_path],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        t1 = time.time()
        process_ms = int((t1 - t0) * 1000)
        if chunks_added == 0:
            return f"未能从文件解析内容：{os.path.basename(file_path)}"      
        return (
            f"文档已添加到知识库：{os.path.basename(file_path)}\n"
            f"分块数量：{chunks_added}\n"
            f"处理时间：{process_ms}ms\n"
            f"命名空间：{pipeline.get('namespace', self.rag_namespace)}")
    
    @tool_action("rag_add_text", "添加文本到知识库")
    def _add_text(
        self,
        text: str,
        document_id: str = None,
        namespace: str = "default",
        chunk_size: int = 800,
        chunk_overlap: int = 100
    ) -> str:
        """添加文本到知识库

        Args:
            text: 要添加的文本内容
            document_id: 文档ID（可选）
            namespace: 知识库命名空间
            chunk_size: 分块大小
            chunk_overlap: 分块重叠大小

        Returns:
            执行结果
        """
        metadata = None
        if not text or not text.strip():
            return "输入的文本内容为空"
        document_id = document_id or f"text_{abs(hash(text)) % 100000}"
        tmp_path = os.path.join(self.knowledge_base_path, f"{document_id}.md")
        with open(tmp_path, 'w', encoding="utf-8", errors="ignore") as f:
            f.write(text)
        pipeline = self._get_pipeline(namespace)
        t0 = time.time()
        chunks_added = pipeline["add_documents"](
            file_paths=[tmp_path],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        t1 = time.time()
        process_ms = int((t1 - t0) * 1000)
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        if chunks_added == 0:
            return f"未能从文本生成有效分块"
        return (
            f"文本已添加到知识库：{document_id}\n"
            f"分块数量：{chunks_added}\n"
            f"处理时间：{process_ms}ms\n"
            f"命名空间：{pipeline.get('namespace', self.rag_namespace)}")
    
    @tool_action("rag_search", "搜索知识库中的相关内容")
    def _search(
        self,
        query: str,
        limit: int = 5,
        min_score: float = 0.1,
        enable_advanced_search: bool = True,
        max_chars: int = 1200,
        include_citations: bool = True,
        namespace: str = "default"
    ) -> str:
        """搜索知识库

        Args:
            query: 搜索查询词
            limit: 返回结果数量
            min_score: 最低相关度分数
            enable_advanced_search: 是否启用高级搜索（MQE、HyDE）
            max_chars: 每个结果最大字符数
            include_citations: 是否包含引用来源
            namespace: 知识库命名空间

        Returns:
            搜索结果
        """
        if not query or not query.strip():
            return "输入的查询为空"
        pipeline = self._get_pipeline(namespace)
        if enable_advanced_search:
            results = pipeline["search_advanced"](
                query=query,
                top_k=limit,
                enable_mqe=True,
                enable_hyde=True,
                score_threshold=min_score if min_score > 0 else None)
        else:
            results = pipeline["search"](
                query=query,
                top_k=limit,
                score_threshold=min_score if min_score > 0 else None)
        if not results:
            return f"未找到与'{query}'相关的内容"
        search_result = ["搜索结果："]
        for i, result in enumerate(results, 1):
            meta = result.get("metadata", {})
            score = result.get("score", 0.0)
            content = meta.get("content", "")[:200] + "..."
            source = meta.get("source_path", "unknown")
            
            def clean_text(text):
                try:
                    return str(text).encode("utf-8", errors="ignore").decode("utf-8")
                except Exception:
                    return str(text)

            clean_content = clean_text(content)
            clean_source = clean_text(source)
            search_result.append(f"\n{i}. 文档：**{clean_source}** （相似度：{score:.3f}）")
            search_result.append(f"   {clean_content}")
            if include_citations and meta.get("heading_path"):
                clean_heading = clean_text(str(meta["heading_path"]))
                search_result.append(f"   章节：{clean_heading}")
        return "\n".join(search_result)
    
    @tool_action("rag_ask", "基于知识库进行智能问答")
    def _ask(
        self,
        question: str,
        limit: int = 5,
        enable_advanced_search: bool = True,
        include_citations: bool = True,
        max_chars: int = 1200,
        namespace: str = "default"
    ) -> str:
        """智能问答：检索 → 上下文注入 → LLM生成答案

        Args:
            question: 用户问题
            limit: 检索结果数量
            enable_advanced_search: 是否启用高级搜索
            include_citations: 是否包含引用来源
            max_chars: 每个结果最大字符数
            namespace: 知识库命名空间

        Returns:
            智能问答结果

        核心流程:
        1. 解析用户问题
        2. 智能检索相关内容
        3. 构建上下文和提示词
        4. LLM生成准确答案
        5. 添加引用来源
        """
        if not question or not question.strip():
            return "输入的问题为空"
        user_question = question.strip()
        print(f"[RAGTool] 智能问答：{user_question}")
        pipeline = self._get_pipeline(namespace)
        search_start = time.time()
        if enable_advanced_search:
            results = pipeline["search_advanced"](
                query=user_question,
                top_k=limit,
                enable_mqe=True,
                enable_hyde=True)
        else:
            results = pipeline["search"](
                query=user_question,
                top_k=limit)
        search_time = int((time.time() - search_start) * 1000)
        if not results:
            return (
                f"抱歉，我在知识库中没有找到与「{user_question}」相关的信息。\n\n"
                f"建议：\n"
                f"• 尝试使用更简洁的关键词\n"
                f"• 检查是否已添加相关文档\n"
                f"• 使用 stats 操作查看知识库状态"
            )
        context_parts = []
        citations = []
        total_score = 0
        for i, result in enumerate(results):
            meta = result.get("metadata", {})
            content = meta.get("content", "").strip()
            source = meta.get("source_path", "unknown")
            score = result.get("score", 0.0)
            total_score += score
            if content:
                cleaned_content = self._clean_content_for_context(content)
                context_parts.append(f"片段 {i+1}：{cleaned_content}")
                if include_citations:
                    citations.append({
                        "index": i+1,
                        "source": os.path.basename(source),
                        "score": score
                    })
        context = "\n\n".join(context_parts)
        if len(context) > max_chars:
            context = self._smart_truncate_context(context, max_chars)
        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(user_question, context)
        enhanced_prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        llm_start = time.time()
        answer = self.llm.invoke(enhanced_prompt)
        llm_time = int((time.time() - llm_start) * 1000)
        if not answer or not answer.strip():
            return "LLM未能生成有效答案，请稍后重试"
        final_answer = self._format_final_answer(
                question=user_question,
                answer=answer.strip(),
                citations=citations if include_citations else None,
                search_time=search_time,
                llm_time=llm_time,
                avg_score=total_score / len(results) if results else 0
            )
        return final_answer

    def _clean_content_for_context(self, content: str) -> str:
        content = " ".join(content.split())
        if len(content) > 300:
            content = content[:300] + "..."
        return content
    
    def _smart_truncate_context(self, context: str, max_chars: int) -> str:
        if len(context) <= max_chars:
            return context
        truncated = context[:max_chars]
        last_break = truncated.rfind("\n\n")
        if last_break > max_chars * 0.7: 
            return truncated[:last_break] + "\n\n[...更多内容被截断]"
        else:
            return truncated[:max_chars-20] + "...[内容被截断]"
    
    def _build_system_prompt(self) -> str:
        return (
            "你是一个专业的知识助手，具备以下能力：\n"
            "1. 精准理解：仔细理解用户问题的核心意图\n"
            "2. 可信回答：严格基于提供的上下文信息回答，不编造内容\n"
            "3. 信息整合：从多个片段中提取关键信息，形成完整答案\n"
            "4. 清晰表达：用简洁明了的语言回答，适当使用结构化格式\n"
            "5. 诚实表达：如果上下文不足以回答问题，请坦诚说明\n\n"
            "回答格式要求：\n"
            "• 直接回答核心问题\n"
            "• 必要时使用要点或步骤\n"
            "• 引用关键原文时使用引号\n"
            "• 避免重复和冗余"
        )
    
    def _build_user_prompt(self, question: str, context: str) -> str:
        return (
            f"请基于以下上下文信息回答问题：\n\n"
            f"【问题】{question}\n\n"
            f"【相关上下文】\n{context}\n\n"
            f"【要求】请提供准确、有帮助的回答。如果上下文信息不足，请说明需要什么额外信息。"
        )
    
    def _format_final_answer(self, question: str, answer: str, citations: Optional[List[Dict]] = None, 
            search_time: int = 0, llm_time: int = 0, avg_score: float = 0) -> str:
        result = [f"**智能问答结果**\n"]
        result.append(answer)
        if citations:
            result.append("\n\n**参考来源**")
            for citation in citations:
                result.append(f"[{citation['index']}] {citation['source']} （相似度: {citation['score']:.3f}）")
        result.append(f"\n===== 检索：{search_time}ms | 生成：{llm_time}ms | 平均相似度：{avg_score:.3f} =====")
        return "\n".join(result)

    @tool_action("rag_clear", "清空知识库（危险操作，请谨慎使用）")
    def _clear_knowledge_base(self, confirm: bool = False, namespace: str = "default") -> str:
        """清空知识库

        Args:
            confirm: 确认执行（必须设置为True）
            namespace: 知识库命名空间

        Returns:
            执行结果
        """
        if not confirm:
            return ("[RAGTool] ⚠️\x20\x20危险操作：清空知识库将删除所有数据！\n"
                    "[RAGTool] 💡\x20请使用 confirm=true 参数确认执行。")
        pipeline = self._get_pipeline(namespace)
        store = pipeline.get("store")
        namespace_id = pipeline.get("namespace", self.rag_namespace)
        success = store.clear_collection() if store else False
        if success:
            self._pipelines[namespace_id] = create_rag_pipeline(
                collection_name=self.collection_name,
                rag_namespace=namespace_id)
            return f"知识库已成功清空（命名空间：{namespace_id}）"
        else:
            return "清空知识库失败"

    @tool_action("rag_stats", "获取知识库统计信息")
    def _get_stats(self, namespace: str = "default") -> str:
        """获取知识库统计

        Args:
            namespace: 知识库命名空间

        Returns:
            统计信息
        """
        pipeline = self._get_pipeline(namespace)
        stats = pipeline["get_stats"]()
        stats_info = [
                "**RAG 知识库统计**",
                f"命名空间：{pipeline.get('namespace', self.rag_namespace)}",
                f"集合名称：{self.collection_name}",
                f"存储根路径：{self.knowledge_base_path}"]
        if stats:
            store_type = stats.get("store_type", "unknown")
            total_vectors = (
                stats.get("points_count") or 
                stats.get("vectors_count") or 
                stats.get("count") or 0
            )
            stats_info.extend([
                f"存储类型：{store_type}",
                f"文档分块数：{int(total_vectors)}",
            ])  
            if "config" in stats:
                config = stats["config"]
                if isinstance(config, dict):
                    vector_size = config.get("vector_size", "unknown")
                    distance = config.get("distance", "unknown")
                    stats_info.extend([
                        f"向量维度：{vector_size}",
                        f"距离度量：{distance}"
                    ])
        stats_info.extend([
            "\n**系统状态**",
            f"RAG 管道：{'正常' if self.initialized else '异常'}",
            f"LLM 连接：{'正常' if hasattr(self, 'llm') else '异常'}"
        ])
        return "\n".join(stats_info)

    def get_relevant_context(self, query: str, limit: int = 3, max_chars: int = 1200, namespace: Optional[str] = None) -> str:
        if not query.strip():
            return ""
        pipeline = self._get_pipeline(namespace)
        results = pipeline["search"](
            query=query,
            top_k=limit
        )
        if not results:
            return ""
        context_parts = []
        for result in results:
            content = result.get("metadata", {}).get("content", "")
            if content:
                context_parts.append(content)
        merged_context = "\n\n".join(context_parts)
        if len(merged_context) > max_chars:
            merged_context = merged_context[:max_chars] + "..."
        return merged_context
    
    def clear_all_namespaces(self) -> str:
        for ns, pipeline in self._pipelines.items():
            store = pipeline.get("store")
            if store:
                store.clear_collection()
        self._pipelines.clear()
        self._init_components()
        return "所有命名空间数据已清空并重新初始化"
    
    def add_document(self, file_path: str, namespace: str = "default") -> str:
        return self.run({
            "action": "add_document",
            "file_path": file_path,
            "namespace": namespace
        })
    
    def add_text(self, text: str, namespace: str = "default", document_id: str = None) -> str:
        return self.run({
            "action": "add_text",
            "text": text,
            "namespace": namespace,
            "document_id": document_id
        })
    
    def ask(self, question: str, namespace: str = "default", **kwargs) -> str:
        params = {
            "action": "ask",
            "question": question,
            "namespace": namespace
        }
        params.update(kwargs)
        return self.run(params)
    
    def search(self, query: str, namespace: str = "default", **kwargs) -> str:
        params = {
            "action": "search",
            "query": query,
            "namespace": namespace
        }
        params.update(kwargs)
        return self.run(params)
    
    def add_documents_batch(self, file_paths: List[str], namespace: str = "default") -> None:
        if not file_paths:
            return "输入的文件路径列表为空"
        for i, file_path in enumerate(file_paths, 1):
            print(f"[RAGTool] ----- 处理文档 {i}/{len(file_paths)}：{os.path.basename(file_path)}")
            result = self.add_document(file_path, namespace)
            print(result)
    
    def add_texts_batch(self, texts: List[str], namespace: str = "default", document_ids: Optional[List[str]] = None) -> str:
        if not texts:
            return "输入的文本列表为空"
        if document_ids and len(document_ids) != len(texts):
            return "文本数量和文档ID数量不匹配"
        for i, text in enumerate(texts):
            doc_id = document_ids[i] if document_ids else f"batch_text_{i+1}"
            print(f"[RAGTool] ----- 处理文本 {i+1}/{len(texts)}：{doc_id}")
            result = self.add_text(text, namespace, doc_id)
            print(result)
