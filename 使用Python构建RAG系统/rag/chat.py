import argparse
import os
import time
from typing import List, Any, Dict, Tuple, Optional

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb


# -------------------------
# Embedding (for query)
# -------------------------

_embedding_model: SentenceTransformer | None = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer("shibing624/text2vec-base-chinese")
    return _embedding_model


def embed_chunk(chunk: str) -> List[float]:
    embedding_model = get_embedding_model()
    embedding = embedding_model.encode(chunk, normalize_embeddings=True)
    return embedding.tolist()


# -------------------------
# Vector store (Chroma)
# -------------------------

def get_chroma_client():
    return chromadb.PersistentClient(path="./chroma_db")


def retrieve(query: str, top_k: int, collection_name: str = "default", include_metadata: bool = False) -> Tuple[List[str], Optional[List[Dict[str, Any]]]]:
    """
    检索相关文档片段
    
    Args:
        query: 查询文本
        top_k: 返回的文档数量
        collection_name: 集合名称
        include_metadata: 是否返回元数据
    
    Returns:
        如果 include_metadata=False: 只返回文档列表
        如果 include_metadata=True: 返回 (文档列表, 元数据列表)
    """
    client = get_chroma_client()
    try:
        collection = client.get_collection(name=collection_name)
    except Exception as e:
        raise RuntimeError(f"集合 '{collection_name}' 不存在，请先运行 build_index.py 构建索引")
    
    query_embedding = embed_chunk(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    documents = results['documents'][0]
    metadatas = results.get('metadatas', [None] * len(documents))[0] if results.get('metadatas') else None
    
    if include_metadata:
        return documents, metadatas
    else:
        return documents, None


def retrieve_simple(query: str, top_k: int, collection_name: str = "default") -> List[str]:
    """简化版检索函数，保持向后兼容性"""
    documents, _ = retrieve(query, top_k, collection_name, include_metadata=False)
    return documents


def extract_images_from_metadata(metadata_list: Optional[List[Dict[str, Any]]]) -> List[List[str]]:
    """从元数据中提取图片URL列表"""
    if not metadata_list:
        return []
    
    images_per_chunk = []
    for metadata in metadata_list:
        if metadata and metadata.get('image_list'):
            image_urls = [url.strip() for url in metadata['image_list'].split(',') if url.strip()]
            images_per_chunk.append(image_urls)
        else:
            images_per_chunk.append([])
    
    return images_per_chunk


def post_process_answer_with_images(answer: str, final_chunks: List[str], final_metadata: Optional[List[Dict[str, Any]]]) -> str:
    """后处理LLM回答，增强图片显示"""
    if not final_metadata:
        return answer
    
    # 收集所有图片
    all_images = []
    for i, (chunk, metadata) in enumerate(zip(final_chunks, final_metadata)):
        if metadata and metadata.get('image_list'):
            image_urls = [url.strip() for url in metadata['image_list'].split(',') if url.strip()]
            for j, url in enumerate(image_urls):
                all_images.append({
                    'chunk_index': i + 1,
                    'image_index': j + 1,
                    'url': url,
                    'title': metadata.get('title', f'片段{i+1}'),
                    'author': metadata.get('nickname', '未知')
                })
    
    if not all_images:
        return answer
    
    # 在回答最后添加图片展示区域
    enhanced_answer = answer
    
    enhanced_answer += "\n\n" + "="*50
    enhanced_answer += "\n📸 相关图片"
    enhanced_answer += "\n" + "="*50
    
    current_chunk = None
    for img in all_images:
        if current_chunk != img['chunk_index']:
            current_chunk = img['chunk_index']
            enhanced_answer += f"\n\n📝 来自: {img['title']} (作者: {img['author']})"
        
        enhanced_answer += f"\n🖼️  图片{img['image_index']}: {img['url']}"
    
    enhanced_answer += "\n" + "="*50
    
    return enhanced_answer


def format_chunks_with_images(chunks: List[str], metadata_list: Optional[List[Dict[str, Any]]]) -> str:
    """格式化文档片段，包含图片信息用于LLM处理"""
    if not metadata_list:
        return "\n\n".join(chunks)
    
    formatted_chunks = []
    for i, (chunk, metadata) in enumerate(zip(chunks, metadata_list)):
        chunk_text = f"片段{i+1}: {chunk}"
        
        if metadata and metadata.get('image_list'):
            image_urls = [url.strip() for url in metadata['image_list'].split(',') if url.strip()]
            if image_urls:
                chunk_text += f"\n[此片段包含{len(image_urls)}张图片]"
                for j, url in enumerate(image_urls[:3], 1):  # 最多包含3张图片
                    chunk_text += f"\n图片{j}: {url}"
                if len(image_urls) > 3:
                    chunk_text += f"\n...还有{len(image_urls)-3}张图片"
        
        formatted_chunks.append(chunk_text)
    
    return "\n\n".join(formatted_chunks)


def format_document_with_metadata(doc: str, metadata: Optional[Dict[str, Any]], index: int) -> str:
    """格式化显示文档和元数据"""
    result = f"[{index}] "
    
    if metadata:
        # 显示标题（如果有）
        if metadata.get('title'):
            result += f"📝 {metadata['title']}\n"
        
        # 显示作者和标签信息
        info_parts = []
        if metadata.get('nickname'):
            info_parts.append(f"👤 {metadata['nickname']}")
        if metadata.get('tag_list'):
            info_parts.append(f"🏷️  {metadata['tag_list']}")
        if metadata.get('liked_count'):
            info_parts.append(f"❤️ {metadata['liked_count']}")
        
        # 显示图片数量
        if metadata.get('image_list'):
            image_count = len([url.strip() for url in metadata['image_list'].split(',') if url.strip()])
            if image_count > 0:
                info_parts.append(f"🖼️ {image_count}张图片")
        
        if info_parts:
            result += f"    {' | '.join(info_parts)}\n"
        
        result += f"    {doc}"
    else:
        result += doc
    
    return result


def display_metadata_details(metadata_list: List[Dict[str, Any]], retrieved_chunks: List[str], title: str = "详细元数据信息") -> None:
    """显示详细的元数据信息"""
    if not metadata_list:
        return
    
    print("\n" + "="*60)
    print(f"📊 {title}")
    print("="*60)
    
    for i, (metadata, chunk) in enumerate(zip(metadata_list, retrieved_chunks)):
        if not metadata:
            continue
            
        print(f"\n📄 文档片段 {i+1}:")
        print("-" * 40)
        
        # 基本信息
        if metadata.get('note_id'):
            print(f"🆔 笔记ID: {metadata['note_id']}")
        if metadata.get('title'):
            print(f"📝 标题: {metadata['title']}")
        if metadata.get('type'):
            print(f"📂 类型: {metadata['type']}")
        if metadata.get('nickname'):
            print(f"👤 作者: {metadata['nickname']}")
        
        # 社交数据
        social_info = []
        if metadata.get('liked_count'):
            social_info.append(f"❤️ 点赞: {metadata['liked_count']}")
        if metadata.get('collected_count'):
            social_info.append(f"💾 收藏: {metadata['collected_count']}")
        if metadata.get('comment_count'):
            social_info.append(f"💬 评论: {metadata['comment_count']}")
        if metadata.get('share_count'):
            social_info.append(f"🔄 分享: {metadata['share_count']}")
        
        if social_info:
            print(f"📊 互动数据: {' | '.join(social_info)}")
        
        # 标签和关键词
        if metadata.get('tag_list'):
            print(f"🏷️  标签: {metadata['tag_list']}")
        if metadata.get('source_keyword'):
            print(f"🔍 来源关键词: {metadata['source_keyword']}")
        
        # 时间信息
        if metadata.get('time'):
            try:
                import datetime
                timestamp = int(metadata['time']) / 1000  # 转换为秒
                date_str = datetime.datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')
                print(f"⏰ 发布时间: {date_str}")
            except:
                print(f"⏰ 时间戳: {metadata['time']}")
        
        # 图片信息
        if metadata.get('image_list'):
            image_list_str = metadata['image_list']
            # 处理可能包含多个图片URL的情况（逗号分隔）
            image_urls = [url.strip() for url in image_list_str.split(',') if url.strip()]
            
            if len(image_urls) == 1:
                print(f"🖼️  图片: {image_urls[0][:60]}{'...' if len(image_urls[0]) > 60 else ''}")
            else:
                print(f"🖼️  图片 ({len(image_urls)}张):")
                for idx, url in enumerate(image_urls[:3], 1):  # 最多显示前3张
                    print(f"    [{idx}] {url[:60]}{'...' if len(url) > 60 else ''}")
                if len(image_urls) > 3:
                    print(f"    ... 还有 {len(image_urls) - 3} 张图片")
        
        # 链接信息
        if metadata.get('note_url'):
            print(f"🔗 链接: {metadata['note_url'][:80]}{'...' if len(metadata['note_url']) > 80 else ''}")
        
        print(f"📄 内容预览: {chunk[:100]}{'...' if len(chunk) > 100 else ''}")
    
    print("="*60)


# -------------------------
# Rerank
# -------------------------

_cross_encoder: CrossEncoder | None = None


def get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
    return _cross_encoder


def rerank(query: str, retrieved_chunks: List[str], top_k: int) -> List[str]:
    if len(retrieved_chunks) <= top_k:
        return retrieved_chunks
    
    cross_encoder = get_cross_encoder()
    pairs = [(query, chunk) for chunk in retrieved_chunks]
    scores = cross_encoder.predict(pairs)

    scored_chunks = list(zip(retrieved_chunks, scores))
    scored_chunks.sort(key=lambda x: x[1], reverse=True)

    return [chunk for chunk, _ in scored_chunks][:top_k]


# -------------------------
# LLM Providers (Gemini & DeepSeek)
# -------------------------

_google_client: Any = None
_deepseek_client: Any = None


def get_google_client():
    global _google_client
    if _google_client is None:
        from google import genai
        load_dotenv()
        _google_client = genai.Client()
    return _google_client


def get_deepseek_client():
    global _deepseek_client
    if _deepseek_client is None:
        load_dotenv()
        from openai import OpenAI
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("缺少 DEEPSEEK_API_KEY 环境变量")
        base_url = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        _deepseek_client = OpenAI(api_key=api_key, base_url=base_url)
    return _deepseek_client


def generate_with_gemini(query: str, chunks: List[str], metadata_list: Optional[List[Dict[str, Any]]] = None) -> str:
    # 先加载环境变量获取 API Key
    load_dotenv()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("缺少 GEMINI_API_KEY 环境变量")
    
    # 为 Gemini 设置临时代理
    original_env = {}
    proxy_config = {
        "HTTP_PROXY": os.environ.get("GEMINI_HTTP_PROXY"),
        "HTTPS_PROXY": os.environ.get("GEMINI_HTTPS_PROXY"),
        "GRPC_PROXY": os.environ.get("GEMINI_HTTPS_PROXY"),
    }
    
    # 备份原始环境变量并设置代理
    # for key, value in proxy_config.items():
    #     if value:
    #         original_env[key] = os.environ.get(key)
    #         os.environ[key] = value
    #         print(f"🔧 为 Gemini 设置代理: {key}={value}")
    
    try:
        # 在代理环境下创建客户端
        from google import genai
        google_client = genai.Client(api_key=api_key)  # 使用当前环境的代理设置
        
        # 格式化包含图片信息的片段
        if metadata_list:
            formatted_chunks = format_chunks_with_images(chunks, metadata_list)
            image_instruction = "\n\n注意：如果相关片段中包含图片，请在回答的最后总结各个片段并加上图片链接"
        else:
            formatted_chunks = "\n\n".join(chunks)
            image_instruction = ""
        
        prompt = f"""你是一位知识助手，请根据用户的问题和下列片段生成准确的回答。

用户问题: {query}

相关片段:
{formatted_chunks}

请基于上述内容作答，不要编造信息。{image_instruction}"""

        print(f"\n===== 提示词（Prompt） =====\n{prompt}\n==========================\n")

        response = google_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )

        return response.text
    finally:
        # 恢复原始环境变量
        for key, original_value in original_env.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value
        
        # 清理新设置的代理环境变量
        for key, value in proxy_config.items():
            if value and key not in original_env:
                os.environ.pop(key, None)


def generate_with_deepseek(query: str, chunks: List[str], metadata_list: Optional[List[Dict[str, Any]]] = None) -> str:
    client = get_deepseek_client()
    
    # 格式化包含图片信息的片段
    if metadata_list:
        formatted_chunks = format_chunks_with_images(chunks, metadata_list)
        image_instruction = "如果相关片段中包含图片，请在回答的适当位置引用这些图片。你可以使用[图片X]的格式来引用图片，并在回答最后列出所有相关图片的链接。"
    else:
        formatted_chunks = "\n\n".join(chunks)
        image_instruction = ""

    system_prompt = f"你是一位知识助手，请依据提供的片段回答问题，避免编造信息。{image_instruction}"
    user_prompt = f"用户问题: {query}\n\n相关片段:\n{formatted_chunks}\n\n请基于上述内容作答，不要编造信息。"

    print(f"\n===== 提示词（Prompt） =====\n[system]\n{system_prompt}\n\n[user]\n{user_prompt}\n==========================\n")

    completion = client.chat.completions.create(
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )
    return completion.choices[0].message.content


def generate(query: str, chunks: List[str], provider: str = "gemini", metadata_list: Optional[List[Dict[str, Any]]] = None) -> str:
    if provider == "deepseek":
        return generate_with_deepseek(query, chunks, metadata_list)
    return generate_with_gemini(query, chunks, metadata_list)


# -------------------------
# CLI
# -------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 问答系统")
    parser.add_argument("--query", type=str, default="哆啦A梦和超级赛亚人的关系", help="查询问题")
    parser.add_argument("--collection", type=str, default="default", help="集合名称（默认: default）")
    parser.add_argument("--top_k_retrieve", type=int, default=2, help="召回条数（默认: 5）")
    parser.add_argument("--top_k_rerank", type=int, default=3, help="重排后保留条数（默认: 3）")
    parser.add_argument("--no_generate", action="store_true", help="只做检索与重排，不进行生成")
    parser.add_argument("--no_rerank", action="store_true", help="跳过重排步骤")
    parser.add_argument("--show_metadata", action="store_true", help="显示详细的元数据信息")
    parser.add_argument("--provider", type=str, default="gemini", choices=["gemini", "deepseek"], help="生成提供方（gemini/deepseek）")

    args = parser.parse_args()

    print(f"查询问题: {args.query}")
    print(f"集合名称: {args.collection}")
    print(f"生成提供方: {args.provider}")

    # 记录总开始时间
    total_start_time = time.time()

    # 1) 检索
    print(f"\n正在检索相关文档片段...")
    retrieve_start_time = time.time()
    try:
        retrieved_chunks, metadata_list = retrieve(args.query, args.top_k_retrieve, args.collection, include_metadata=True)
    except RuntimeError as e:
        print(f"❌ 错误: {e}")
        return
    retrieve_end_time = time.time()
    retrieve_time = retrieve_end_time - retrieve_start_time

    print(f"✅ 检索完成，耗时: {retrieve_time:.3f}秒")
    print(f"检索到 {len(retrieved_chunks)} 个相关片段")

    if not retrieved_chunks:
        print("❌ 未找到相关文档片段")
        return

    # 检查是否有元数据
    has_metadata = metadata_list and any(meta for meta in metadata_list)

    print("\n===== 初步召回结果 =====")
    for i, chunk in enumerate(retrieved_chunks):
        if has_metadata and i < len(metadata_list):
            formatted_result = format_document_with_metadata(chunk, metadata_list[i], i)
            print(f"{formatted_result}\n")
        else:
            print(f"[{i}] {chunk}\n")

    # 显示详细元数据（如果请求）
    if args.show_metadata and has_metadata:
        display_metadata_details(metadata_list, retrieved_chunks, "初步召回详细元数据信息")

    # 2) 重排（可选）
    if args.no_rerank:
        final_chunks = retrieved_chunks[:args.top_k_rerank]
        print("⏩ 跳过重排步骤")
        rerank_time = 0
    else:
        print("正在重排...")
        rerank_start_time = time.time()
        final_chunks = rerank(args.query, retrieved_chunks, args.top_k_rerank)
        rerank_end_time = time.time()
        rerank_time = rerank_end_time - rerank_start_time

        print(f"✅ 重排完成，耗时: {rerank_time:.3f}秒")
        print("===== 重排后结果 =====")
        for i, chunk in enumerate(final_chunks):
            # 找到原始元数据
            original_index = retrieved_chunks.index(chunk) if chunk in retrieved_chunks else -1
            metadata = None
            if has_metadata and original_index >= 0 and original_index < len(metadata_list):
                metadata = metadata_list[original_index]
            
            if metadata:
                formatted_result = format_document_with_metadata(chunk, metadata, i)
                print(f"{formatted_result}\n")
            else:
                print(f"[{i}] {chunk}\n")

        # 显示重排后的详细元数据（如果请求）
        if args.show_metadata and has_metadata:
            # 构建重排后的元数据列表
            reranked_metadata = []
            for chunk in final_chunks:
                original_index = retrieved_chunks.index(chunk) if chunk in retrieved_chunks else -1
                if original_index >= 0 and original_index < len(metadata_list):
                    reranked_metadata.append(metadata_list[original_index])
                else:
                    reranked_metadata.append({})
            
            display_metadata_details(reranked_metadata, final_chunks, "重排后详细元数据信息")

    # 3) 生成（可选）
    if args.no_generate:
        print("⏩ 跳过生成步骤")
        generate_time = 0
    else:
        print(f"正在使用 {args.provider} 生成回答...")
        generate_start_time = time.time()
        try:
            # 构建最终块对应的元数据
            final_metadata = []
            if has_metadata:
                for chunk in final_chunks:
                    original_index = retrieved_chunks.index(chunk) if chunk in retrieved_chunks else -1
                    if original_index >= 0 and original_index < len(metadata_list):
                        final_metadata.append(metadata_list[original_index])
                    else:
                        final_metadata.append({})
            else:
                final_metadata = None
            
            answer = generate(args.query, final_chunks, provider=args.provider, metadata_list=final_metadata)
            
            # 后处理回答，增强图片显示
            # enhanced_answer = post_process_answer_with_images(answer, final_chunks, final_metadata)
            
            generate_end_time = time.time()
            generate_time = generate_end_time - generate_start_time
            
            print(f"✅ 生成完成，耗时: {generate_time:.3f}秒")
            print("===== 最终回答 =====")
            print(answer)
        except Exception as e:
            generate_end_time = time.time()
            generate_time = generate_end_time - generate_start_time
            print(f"❌ 生成回答时出错（耗时: {generate_time:.3f}秒）: {e}")
            return

    # 计算并显示总耗时
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    print("\n" + "="*50)
    print("📊 性能统计")
    print("="*50)
    print(f"🔍 检索耗时: {retrieve_time:.3f}秒")
    print(f"🔄 重排耗时: {rerank_time:.3f}秒")
    if not args.no_generate:
        print(f"✨ 生成耗时: {generate_time:.3f}秒")
    print(f"⏱️  总耗时: {total_time:.3f}秒")
    print("="*50)


if __name__ == "__main__":
    main() 