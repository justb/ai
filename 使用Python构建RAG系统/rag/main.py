import argparse
import os
from typing import List, Any

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder
import chromadb


# -------------------------
# Data preparation
# -------------------------

def split_into_chunks(doc_file: str) -> List[str]:
    with open(doc_file, 'r', encoding='utf-8') as file:
        content = file.read()
    return [chunk for chunk in content.split("\n\n")]


# -------------------------
# Embedding
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

_chroma_client: Any = None
_chroma_collection: Any = None


def get_chroma_collection():
    global _chroma_client, _chroma_collection
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(path="./chroma_db")
    if _chroma_collection is None:
        _chroma_collection = _chroma_client.get_or_create_collection(name="default")
    return _chroma_collection


def save_embeddings(chunks: List[str], embeddings: List[List[float]]) -> None:
    collection = get_chroma_collection()
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[str(i)]
        )


# -------------------------
# Retrieval & Rerank
# -------------------------

_cross_encoder: CrossEncoder | None = None


def get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder('cross-encoder/mmarco-mMiniLMv2-L12-H384-v1')
    return _cross_encoder


def retrieve(query: str, top_k: int) -> List[str]:
    collection = get_chroma_collection()
    query_embedding = embed_chunk(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    return results['documents'][0]


def rerank(query: str, retrieved_chunks: List[str], top_k: int) -> List[str]:
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
    google_client = get_google_client()

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
    parser = argparse.ArgumentParser(description="Simple RAG pipeline with selectable provider (Gemini or DeepSeek)")
    parser.add_argument("--doc", type=str, default="doc.md", help="文档路径（默认: doc.md）")
    parser.add_argument("--query", type=str, default="哆啦A梦使用的3个秘密道具分别是什么？", help="查询问题")
    parser.add_argument("--top_k_retrieve", type=int, default=5, help="召回条数（默认: 5）")
    parser.add_argument("--top_k_rerank", type=int, default=3, help="重排后保留条数（默认: 3）")
    parser.add_argument("--no_generate", action="store_true", help="只做检索与重排，不进行生成")
    parser.add_argument("--provider", type=str, default="gemini", choices=["gemini", "deepseek"], help="生成提供方（gemini/deepseek）")

    args = parser.parse_args()

    if not os.path.exists(args.doc):
        raise FileNotFoundError(f"未找到文档文件: {args.doc}")

    # 1) 切分
    chunks = split_into_chunks(args.doc)
    print(f"已切分片段数: {len(chunks)}")

    # 2) 向量化
    embeddings = [embed_chunk(chunk) for chunk in chunks]
    print(f"已生成向量数: {len(embeddings)}")

    # 3) 存入向量库
    save_embeddings(chunks, embeddings)

    # 4) 检索
    retrieved_chunks = retrieve(args.query, args.top_k_retrieve)

    print("\n===== 初步召回结果 =====")
    for i, chunk in enumerate(retrieved_chunks):
        print(f"[{i}] {chunk}\n")

    # 5) 重排
    reranked_chunks = rerank(args.query, retrieved_chunks, args.top_k_rerank)

    print("===== 重排后结果 =====")
    for i, chunk in enumerate(reranked_chunks):
        print(f"[{i}] {chunk}\n")

    # 6) 生成
    if args.no_generate:
        return

    answer = generate(args.query, reranked_chunks, provider=args.provider)

    print("===== 最终回答 =====")
    print(answer)


if __name__ == "__main__":
    main() 