import os
import re
import json
import time
import hashlib
import sqlite3
from typing import List, Dict, Optional, Any
from core import OpenAICompatibleLLM
from ..embedding import get_text_embedder, get_dimension
from ..storage.qdrant_store import QdrantVectorStore, QdrantConnectionManager

def _fallback_text_reader(path: str) -> str:
    with open(path, 'r', encoding="utf-8", errors="ignore") as f:
        return f.read()

def _convert_to_markdown(path: str) -> str:
    from markitdown import MarkItDown
    md = MarkItDown()
    if md is None:
        return _fallback_text_reader(path)
    try:
        result = md.convert(path)
    except Exception as e:
        print(f"[MarkItDown] ⚠️\x20\x20转换文件失败，自动回退到纯文本读取方式：\n{str(e)}")
        return _fallback_text_reader(path)
    text = getattr(result, "text_content", None)
    if not text or not text.strip():
        return ""
    ext = (os.path.splitext(path)[1] or '').lower()
    if ext == ".pdf":
        return _post_process_pdf_text(text)
    return text

def _post_process_pdf_text(text: str) -> str:
    lines = text.splitlines()
    cleaned_lines = []
    for line in lines:
        line = line.strip()
        if not line:
            continue  
        # 移除单个字符的行（通常是噪音）
        if len(line) <= 2 and not line.isdigit():
            continue
        # 移除明显的页眉页脚噪音
        if re.match(r'^\d+$', line):
            continue
        if line.lower() in ["github", "project", "forks", "stars", "language"]:
            continue
        cleaned_lines.append(line)
    merged_lines = []
    i = 0
    while i < len(cleaned_lines):
        current_line = cleaned_lines[i]    
        # 如果当前行很短，尝试与下一行合并
        if len(current_line) < 60 and i + 1 < len(cleaned_lines):
            next_line = cleaned_lines[i + 1]
            # 合并条件：都是内容，不是标题
            if (not current_line.endswith('：') and 
                not current_line.endswith(':') and
                not current_line.startswith('#') and
                not next_line.startswith('#') and
                len(next_line) < 120):      
                merged_line = current_line + " " + next_line
                merged_lines.append(merged_line)
                # 跳过下一行
                i += 2
                continue
        merged_lines.append(current_line)
        i += 1
    paragraphs = []
    current_paragraph = []
    for line in merged_lines:
        # 检查是否是新段落的开始
        if (line.startswith('#') or  # 标题
            line.endswith('：') or   # 中文冒号结尾
            line.endswith(':') or    # 英文冒号结尾
            len(line) > 150 or       # 长句通常是段落开始
            not current_paragraph):  # 第一行
            # 保存当前段落
            if current_paragraph:
                paragraphs.append(' '.join(current_paragraph))
                current_paragraph = []
            paragraphs.append(line)
        else:
            current_paragraph.append(line)
    # 添加最后一个段落
    if current_paragraph:
        paragraphs.append(' '.join(current_paragraph))
    return "\n\n".join(paragraphs)

def _detect_lang(sample: str) -> str:
    try:
        from langdetect import detect
        return detect(sample[:1000]) if sample else "unknown"
    except Exception:
        return "unknown"

def _is_cjk(ch: str) -> bool:
    code = ord(ch)
    return (
        0x4E00 <= code <= 0x9FFF or
        0x3400 <= code <= 0x4DBF or
        0x20000 <= code <= 0x2A6DF or
        0x2A700 <= code <= 0x2B73F or
        0x2B740 <= code <= 0x2B81F or
        0x2B820 <= code <= 0x2CEAF or
        0xF900 <= code <= 0xFAFF
    )

def _approx_token_len(text: str) -> int:
    # 近似估算：cjk字符算作一个token，其他按空白分词
    cjk_tokens = sum(1 for ch in text if _is_cjk(ch))
    non_cjk_tokens = len([t for t in text.split() if t])
    return cjk_tokens + non_cjk_tokens

def _split_paragraphs_with_headings(text: str) -> List[Dict]:
    lines = text.splitlines()
    heading_stack: List[str] = []
    paragraphs: List[Dict] = []
    buf: List[str] = []
    char_pos = 0
    def flush_buf(end_pos: int):
        if not buf:
            return
        content = "\n".join(buf).strip()
        if not content:
            return
        paragraphs.append({
            "content": content,
            "heading_path": " > ".join(heading_stack) if heading_stack else None,
            "start": max(0, end_pos - len(content)),
            "end": end_pos,
        })
    for ln in lines:
        raw = ln
        if raw.strip().startswith('#'):
            flush_buf(char_pos)
            level = len(raw) - len(raw.lstrip('#'))
            title = raw.lstrip('#').strip()
            if level <= 0:
                level = 1
            if level <= len(heading_stack):
                heading_stack = heading_stack[:level-1]
            heading_stack.append(title)
            char_pos += len(raw) + 1
            continue
        if raw.strip() == "":
            flush_buf(char_pos)
            buf = []
        else:
            buf.append(raw)
        char_pos += len(raw) + 1
    flush_buf(char_pos)
    if not paragraphs:
        paragraphs = [{"content": text, "heading_path": None, "start": 0, "end": len(text)}]
    return paragraphs

def _chunk_paragraphs(paragraphs: List[Dict], chunk_tokens: int, overlap_tokens: int) -> List[Dict]:
    chunks: List[Dict] = []
    cur: List[Dict] = []
    cur_tokens = 0
    i = 0
    while i < len(paragraphs):
        p = paragraphs[i]
        p_tokens = _approx_token_len(p["content"]) or 1
        if cur_tokens + p_tokens <= chunk_tokens or not cur:
            cur.append(p)
            cur_tokens += p_tokens
            i += 1
        else:
            content = "\n\n".join(x["content"] for x in cur)
            start = cur[0]["start"]
            end = cur[-1]["end"]
            heading_path = next((x["heading_path"] for x in reversed(cur) if x.get("heading_path")), None)
            chunks.append({
                "content": content,
                "start": start,
                "end": end,
                "heading_path": heading_path,
            })
            if overlap_tokens > 0 and cur:
                kept: List[Dict] = []
                kept_tokens = 0
                for x in reversed(cur):
                    t = _approx_token_len(x["content"]) or 1
                    if kept_tokens + t > overlap_tokens:
                        break
                    kept.append(x)
                    kept_tokens += t
                cur = list(reversed(kept))
                cur_tokens = kept_tokens
            else:
                cur = []
                cur_tokens = 0
    if cur:
        content = "\n\n".join(x["content"] for x in cur)
        start = cur[0]["start"]
        end = cur[-1]["end"]
        heading_path = next((x["heading_path"] for x in reversed(cur) if x.get("heading_path")), None)
        chunks.append({
            "content": content,
            "start": start,
            "end": end,
            "heading_path": heading_path,
        })
    return chunks

def load_and_chunk_texts(
        paths: List[str], 
        chunk_size: int, 
        chunk_overlap: int, 
        namespace: str, 
        source_label: str = "rag"
    ) -> List[Dict]:
    print(f"[RAG] 开始文本分块：files={len(paths)}, chunk_size={chunk_size}, overlap={chunk_overlap}, namespace={namespace}")
    chunks: List[Dict] = []
    seen_hashes = set()
    for path in paths:
        if not os.path.exists(path):
            print(f"[RAG] ⚠️\x20\x20文件不存在：{path}")
            continue
        ext = (os.path.splitext(path)[1] or '').lower()
        markdown_text = _convert_to_markdown(path)
        if not markdown_text.strip():
            print(f"[RAG] ⚠️\x20\x20提取的Markdown文本为空： {path}")
            continue
        lang = _detect_lang(markdown_text)
        doc_id = hashlib.md5(f"{path}|{len(markdown_text)}".encode("utf-8")).hexdigest()
        para = _split_paragraphs_with_headings(markdown_text)
        token_chunks = _chunk_paragraphs(para, chunk_tokens=max(1, chunk_size), overlap_tokens=max(0, chunk_overlap))
        for ch in token_chunks:
            content = ch["content"]
            start = ch.get("start", 0)
            end = ch.get("end", start + len(content))
            norm = content.strip()
            if not norm:
                continue
            content_hash = hashlib.md5(norm.encode("utf-8")).hexdigest()
            if content_hash in seen_hashes:
                continue
            seen_hashes.add(content_hash)
            chunk_id = hashlib.md5(f"{doc_id}|{start}|{end}|{content_hash}".encode("utf-8")).hexdigest()
            chunks.append({
                "id": chunk_id,
                "content": content,
                "metadata": {
                    "source_path": path,
                    "file_ext": ext,
                    "doc_id": doc_id,
                    "lang": lang,
                    "start": start,
                    "end": end,
                    "content_hash": content_hash,
                    "namespace": namespace or "default",
                    "source": source_label,
                    "external": True,
                    "heading_path": ch.get("heading_path"),
                    "format": "markdown",
                },
            })
    print(f"[RAG] 文本分块完成：total_chunks={len(chunks)}")
    return chunks

def _preprocess_markdown_for_embedding(text: str) -> str:
    # 去除标题符号，并保留文本
    text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
    # 去除网页链接，并保留文本
    text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
    # 去除粗体、斜体、内联符号
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text) 
    text = re.sub(r'\*([^*]+)\*', r'\1', text)     
    text = re.sub(r'`([^`]+)`', r'\1', text)     
    # 去除代码块标识符，并保留文本
    text = re.sub(r'```[^\n]*\n([\s\S]*?)```', r'\1', text)
    # 去除制表符、换行符、多余空白
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()

def index_chunks(
    store, 
    chunks: List[Dict],
    cache_db: Optional[str] = None, 
    batch_size: int = 64,
    namespace: str = "default"
) -> None:
    if not chunks:
        return
    embedder = get_text_embedder()
    dimension = get_dimension()
    processed_texts = []
    for c in chunks:
        raw_content = c["content"]
        processed_content = _preprocess_markdown_for_embedding(raw_content)
        processed_texts.append(processed_content)
    print(f"[RAG] 开始转换为嵌入向量：total_texts={len(processed_texts)}, batch_size={batch_size}")
    vecs: List[List[float]] = []
    for i in range(0, len(processed_texts), batch_size):
        part = processed_texts[i:i+batch_size]
        try:
            part_vecs = embedder.encode(part)
            if not isinstance(part_vecs, list):
                if hasattr(part_vecs, "tolist"):
                    part_vecs = [part_vecs.tolist()]
                else:
                    part_vecs = [list(part_vecs)]
            else:
                if part_vecs and not isinstance(part_vecs[0], (list, tuple)) and hasattr(part_vecs[0], "__len__"):
                    normalized_vecs = []
                    for v in part_vecs:
                        if hasattr(v, "tolist"):
                            normalized_vecs.append(v.tolist())
                        else:
                            normalized_vecs.append(list(v))
                    part_vecs = normalized_vecs
                elif part_vecs and not isinstance(part_vecs[0], (list, tuple)):
                    if hasattr(part_vecs, "tolist"):
                        part_vecs = [part_vecs.tolist()]
                    else:
                        part_vecs = [list(part_vecs)]
            for v in part_vecs:
                try:
                    if hasattr(v, "tolist"):
                        v = v.tolist()
                    v_norm = [float(x) for x in v]
                    if len(v_norm) != dimension:
                        print(f"[RAG] ⚠️\x20\x20向量维度异常：期望{dimension}, 实际{len(v_norm)}")
                        if len(v_norm) < dimension:
                            v_norm.extend([0.0] * (dimension - len(v_norm)))
                        else:
                            v_norm = v_norm[:dimension]
                    vecs.append(v_norm)
                except Exception as e:
                    print(f"[RAG] ⚠️\x20\x20向量转换失败, 使用零向量： {str(e)}")
                    vecs.append([0.0] * dimension)
        except Exception as e:
            print(f"[RAG] ⚠️\x20\x20Batch {i} 编码失败，尝试更小的分块：{str(e)}")
            success = False
            for j in range(0, len(part), 8):
                small_part = part[j:j+8]
                try:
                    # 等待2秒，避免Embedding API访问限制
                    import time
                    time.sleep(2)
                    small_vecs = embedder.encode(small_part)
                    if isinstance(small_vecs, list) and small_vecs and not isinstance(small_vecs[0], list):
                        small_vecs = [small_vecs]
                    for v in small_vecs:
                        if hasattr(v, "tolist"):
                            v = v.tolist()
                        try:
                            v_norm = [float(x) for x in v]
                            if len(v_norm) != dimension:
                                print(f"[RAG] ⚠️\x20\x20向量维度异常：期望{dimension}, 实际{len(v_norm)}")
                                if len(v_norm) < dimension:
                                    v_norm.extend([0.0] * (dimension - len(v_norm)))
                                else:
                                    v_norm = v_norm[:dimension]
                            vecs.append(v_norm)
                            success = True
                        except Exception as e2:
                            print(f"[RAG] ⚠️\x20\x20小批次向量转换失败： {str(e2)}")
                            vecs.append([0.0] * dimension)
                except Exception as e2:
                    print(f"[RAG] ⚠️\x20\x20小批次 {j//8} 仍然失败：{str(e2)}")
                    for _ in range(len(small_part)):
                        vecs.append([0.0] * dimension)
            if not success:
                print(f"[RAG] ⛔\x20批次 {i} 完全失败, 使用零向量")
        print(f"[RAG] 嵌入向量转换进度：{min(i+batch_size, len(processed_texts))}/{len(processed_texts)}")
    metas: List[Dict] = []
    ids: List[str] = []
    for ch in chunks:
        meta = {
            "memory_id": ch["id"],
            "user_id": "rag_user",
            "memory_type": "rag_chunk",
            "content": ch["content"],
            "data_source": "rag_pipeline",
            "rag_namespace": namespace,
            "is_rag_data": True,
        }
        meta.update(ch.get("metadata", {}))
        metas.append(meta)
        ids.append(ch["id"])
    success = store.add_vectors(vectors=vecs, metadata=metas, ids=ids)
    if not success:
        raise RuntimeError("存入向量数据库失败")

def embed_query(query: str) -> List[float]:
    embedder = get_text_embedder()
    dimension = get_dimension()
    vec = embedder.encode(query)
    if hasattr(vec, "tolist"):
        vec = vec.tolist()
    if isinstance(vec, list) and vec and isinstance(vec[0], (list, tuple)):
        vec = vec[0]
    result = [float(x) for x in vec]
    if len(result) != dimension:
        print(f"[RAG] ⚠️\x20\x20查询向量维度异常：期望{dimension}, 实际{len(result)}")
        if len(result) < dimension:
            result.extend([0.0] * (dimension - len(result)))
        else:
            result = result[:dimension]
    return result

def search_vectors(
    store: QdrantVectorStore,
    query: str, 
    top_k: int = 8, 
    rag_namespace: Optional[str] = None, 
    only_rag_data: bool = True, 
    score_threshold: Optional[float] = None
) -> List[Dict]:
    if not query.strip():
        return []
    qv = embed_query(query)
    where = {"memory_type": "rag_chunk"}
    if only_rag_data:
        where["is_rag_data"] = True
        where["data_source"] = "rag_pipeline"
    if rag_namespace:
        where["rag_namespace"] = rag_namespace
    return store.search_similar(
        query_vector=qv, 
        limit=top_k, 
        score_threshold=score_threshold, 
        where=where
    )

def _prompt_mqe(llm: OpenAICompatibleLLM, query: str, n: int) -> List[str]:
    prompt = [
        {"role": "system", "content": "你是检索查询扩展助手。生成语义等价或互补的多样化查询。使用中文，简短，避免标点。"},
        {"role": "user", "content": f"原始查询：{query}\n请给出{n}个不同表述的查询，每行一个。"}
    ]
    text = llm.invoke(prompt)
    lines = [ln.strip("- \t") for ln in (text or "").splitlines()]
    outs = [ln for ln in lines if ln]
    return outs[:n] or [query]

def _prompt_hyde(llm: OpenAICompatibleLLM, query: str) -> Optional[str]:
    prompt = [
        {"role": "system", "content": "根据用户问题，先写一段可能的答案性段落，用于向量检索的查询文档（不要分析过程）。"},
        {"role": "user", "content": f"问题：{query}\n请直接写一段中等长度、客观、包含关键术语的段落。"}
    ]
    return llm.invoke(prompt)

def search_vectors_expanded(
    store: QdrantVectorStore,
    llm: OpenAICompatibleLLM,
    query: str,
    top_k: int = 8,
    rag_namespace: Optional[str] = None,
    only_rag_data: bool = True,
    score_threshold: Optional[float] = None,
    enable_mqe: bool = True,
    mqe_expansions: int = 2,
    enable_hyde: bool = False,
    candidate_pool_multiplier: int = 4,
) -> List[Dict]:
    if not query.strip():
        return []
    expansions: List[str] = [query]
    if enable_mqe and mqe_expansions > 0:
        expansions.extend(_prompt_mqe(llm, query, mqe_expansions))
    if enable_hyde:
        hyde_text = _prompt_hyde(llm, query)
        if hyde_text:
            expansions.append(hyde_text)
    uniq: List[str] = []
    for e in expansions:
        if e and e not in uniq:
            uniq.append(e)
    expansions = uniq[: max(1, len(uniq))]
    pool = max(top_k * candidate_pool_multiplier, 20)
    per = max(1, pool // max(1, len(expansions)))
    where = {"memory_type": "rag_chunk"}
    if only_rag_data:
        where["is_rag_data"] = True
        where["data_source"] = "rag_pipeline"
    if rag_namespace:
        where["rag_namespace"] = rag_namespace
    agg: Dict[str, Dict] = {}
    for q in expansions:
        qv = embed_query(q)
        hits = store.search_similar(query_vector=qv, limit=per, score_threshold=score_threshold, where=where)
        for h in hits:
            mid = h.get("metadata", {}).get("memory_id", h.get("id"))
            s = float(h.get("score", 0.0))
            if mid not in agg or s > float(agg[mid].get("score", 0.0)):
                agg[mid] = h
    merged = list(agg.values())
    merged.sort(key=lambda x: float(x.get("score", 0.0)), reverse=True)
    return merged[:top_k]

def create_rag_pipeline(
    collection_name: str = "hello_agents_rag_vectors",
    rag_namespace: str = "default"
) -> Dict[str, Any]:
    store = QdrantConnectionManager.get_instance(collection_name=collection_name)
    llm = OpenAICompatibleLLM()

    def add_documents(file_paths: List[str], chunk_size: int = 800, chunk_overlap: int = 100):
        chunks = load_and_chunk_texts(
            paths=file_paths,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            namespace=rag_namespace,
            source_label="rag"
        )
        index_chunks(
            store=store,
            chunks=chunks,
            namespace=rag_namespace
        )
        return len(chunks)
    
    def search(query: str, top_k: int = 8, score_threshold: Optional[float] = None):
        return search_vectors(
            store=store,
            query=query,
            top_k=top_k,
            rag_namespace=rag_namespace,
            score_threshold=score_threshold)
    
    def search_advanced(
        query: str, 
        top_k: int = 8, 
        enable_mqe: bool = True,
        enable_hyde: bool = False,
        score_threshold: Optional[float] = None):
        return search_vectors_expanded(
            store=store,
            llm=llm,
            query=query,
            top_k=top_k,
            rag_namespace=rag_namespace,
            enable_mqe=enable_mqe,
            enable_hyde=enable_hyde,
            score_threshold=score_threshold
        )
    
    def get_stats():
        return store.get_collection_stats()
    
    return {
        "store": store,
        "namespace": rag_namespace,
        "add_documents": add_documents,
        "search": search,
        "search_advanced": search_advanced,
        "get_stats": get_stats
    }