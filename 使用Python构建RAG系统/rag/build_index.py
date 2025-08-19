import argparse
import os
from typing import List, Any

from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
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

def get_chroma_client():
    return chromadb.PersistentClient(path="./chroma_db")


def save_embeddings(chunks: List[str], embeddings: List[List[float]], collection_name: str = "default") -> None:
    client = get_chroma_client()
    collection = client.get_or_create_collection(name=collection_name)
    
    # 清空现有数据（重新构建索引）
    try:
        client.delete_collection(name=collection_name)
        collection = client.create_collection(name=collection_name)
        print(f"已清空并重新创建集合: {collection_name}")
    except Exception:
        collection = client.get_or_create_collection(name=collection_name)
        print(f"使用现有集合: {collection_name}")
    
    for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[str(i)]
        )


# -------------------------
# CLI
# -------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="构建向量索引")
    parser.add_argument("--doc", type=str, required=True, help="文档路径")
    parser.add_argument("--collection", type=str, default="default", help="集合名称（默认: default）")
    parser.add_argument("--chunk_method", type=str, default="paragraph", choices=["paragraph", "sentence"], help="分块方法（默认: paragraph）")

    args = parser.parse_args()

    if not os.path.exists(args.doc):
        raise FileNotFoundError(f"未找到文档文件: {args.doc}")

    print(f"正在处理文档: {args.doc}")
    print(f"集合名称: {args.collection}")
    print(f"分块方法: {args.chunk_method}")

    # 1) 切分
    chunks = split_into_chunks(args.doc)
    print(f"已切分片段数: {len(chunks)}")

    if len(chunks) == 0:
        print("警告: 未找到任何文档片段")
        return

    # 显示前几个片段的预览
    print("\n===== 文档片段预览 =====")
    for i, chunk in enumerate(chunks[:3]):
        print(f"[{i+1}] {chunk[:100]}{'...' if len(chunk) > 100 else ''}\n")

    # 2) 向量化
    print("正在生成向量嵌入...")
    embeddings = [embed_chunk(chunk) for chunk in chunks]
    print(f"已生成向量数: {len(embeddings)}")
    print(f"向量维度: {len(embeddings[0]) if embeddings else 0}")

    # 3) 存入向量库
    print("正在保存到向量数据库...")
    save_embeddings(chunks, embeddings, args.collection)

    print(f"✅ 向量索引构建完成！")
    print(f"   - 文档片段: {len(chunks)} 个")
    print(f"   - 向量维度: {len(embeddings[0]) if embeddings else 0}")
    print(f"   - 存储位置: ./chroma_db")
    print(f"   - 集合名称: {args.collection}")


if __name__ == "__main__":
    main() 