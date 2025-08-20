import argparse
import os
import time
from typing import List, Any

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


def retrieve(query: str, top_k: int, collection_name: str = "default") -> List[str]:
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
    return results['documents'][0]


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


def generate_with_gemini(query: str, chunks: List[str]) -> str:
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
        joined_chunks = "\n\n".join(chunks)
        prompt = f"""你是一位知识助手，请根据用户的问题和下列片段生成准确的回答。

用户问题: {query}

相关片段:
{joined_chunks}

请基于上述内容作答，不要编造信息。"""

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


def generate_with_deepseek(query: str, chunks: List[str]) -> str:
    client = get_deepseek_client()
    joined_chunks = "\n\n".join(chunks)

    system_prompt = "你是一位知识助手，请依据提供的片段回答问题，避免编造信息。"
    user_prompt = f"用户问题: {query}\n\n相关片段:\n{joined_chunks}\n\n请基于上述内容作答，不要编造信息。"

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


def generate(query: str, chunks: List[str], provider: str = "gemini") -> str:
    if provider == "deepseek":
        return generate_with_deepseek(query, chunks)
    return generate_with_gemini(query, chunks)


# -------------------------
# CLI
# -------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="RAG 问答系统")
    parser.add_argument("--query", type=str, default="哆啦A梦和超级赛亚人的关系", help="查询问题")
    parser.add_argument("--collection", type=str, default="default", help="集合名称（默认: default）")
    parser.add_argument("--top_k_retrieve", type=int, default=5, help="召回条数（默认: 5）")
    parser.add_argument("--top_k_rerank", type=int, default=3, help="重排后保留条数（默认: 3）")
    parser.add_argument("--no_generate", action="store_true", help="只做检索与重排，不进行生成")
    parser.add_argument("--no_rerank", action="store_true", help="跳过重排步骤")
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
        retrieved_chunks = retrieve(args.query, args.top_k_retrieve, args.collection)
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

    print("\n===== 初步召回结果 =====")
    for i, chunk in enumerate(retrieved_chunks):
        print(f"[{i}] {chunk}\n")

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
            print(f"[{i}] {chunk}\n")

    # 3) 生成（可选）
    if args.no_generate:
        print("⏩ 跳过生成步骤")
        generate_time = 0
    else:
        print(f"正在使用 {args.provider} 生成回答...")
        generate_start_time = time.time()
        try:
            answer = generate(args.query, final_chunks, provider=args.provider)
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