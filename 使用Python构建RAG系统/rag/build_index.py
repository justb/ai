import argparse
import json
import os
import time
from typing import List, Any, Dict

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


def load_json_data(json_file: str) -> List[Dict[str, Any]]:
    """加载 JSON 文件数据"""
    with open(json_file, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data


def process_json_to_chunks(json_data: List[Dict[str, Any]]) -> tuple[List[str], List[Dict[str, Any]]]:
    """
    将 JSON 数据转换为文档块和元数据
    返回: (文档内容列表, 元数据列表)
    """
    chunks = []
    metadata_list = []
    
    for item in json_data:
        # 组合标题和描述作为文档内容
        title = item.get('title', '').strip()
        desc = item.get('desc', '').strip()
        
        # 跳过空内容
        if not title and not desc:
            continue
            
        # 组合文档内容
        content_parts = []
        if title:
            content_parts.append(f"标题: {title}")
        if desc:
            content_parts.append(f"内容: {desc}")
        
        document_content = "\n".join(content_parts)
        chunks.append(document_content)
        
        # 保存元数据
        metadata = {
            'note_id': item.get('note_id', ''),
            'type': item.get('type', ''),
            'title': title,
            'nickname': item.get('nickname', ''),
            'tag_list': item.get('tag_list', ''),
            'source_keyword': item.get('source_keyword', ''),
            'image_list': item.get('image_list', ''),
            'liked_count': item.get('liked_count', '0'),
            'collected_count': item.get('collected_count', '0'),
            'comment_count': item.get('comment_count', '0'),
            'share_count': item.get('share_count', '0'),
            'note_url': item.get('note_url', ''),
            'time': item.get('time', 0)
        }
        metadata_list.append(metadata)
    
    return chunks, metadata_list


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


def save_embeddings(chunks: List[str], embeddings: List[List[float]], collection_name: str = "default", metadata_list: List[Dict[str, Any]] = None) -> None:
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
        # 准备元数据
        metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else {}
        
        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[str(i)]
        )


# -------------------------
# CLI
# -------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="构建向量索引")
    parser.add_argument("--input", type=str, required=True, help="输入文件路径（支持 .txt 或 .json 文件）")
    parser.add_argument("--collection", type=str, default="default", help="集合名称（默认: default）")
    parser.add_argument("--chunk_method", type=str, default="paragraph", choices=["paragraph", "sentence"], help="分块方法（默认: paragraph，仅适用于文本文件）")

    args = parser.parse_args()

    if not os.path.exists(args.input):
        raise FileNotFoundError(f"未找到输入文件: {args.input}")

    print(f"正在处理文件: {args.input}")
    print(f"集合名称: {args.collection}")
    
    # 记录开始时间
    start_time = time.time()
    
    # 检测文件类型并处理
    file_extension = os.path.splitext(args.input)[1].lower()
    metadata_list = None
    
    if file_extension == '.json':
        print("文件类型: JSON")
        print("正在解析 JSON 数据...")
        
        # 处理 JSON 文件
        json_data = load_json_data(args.input)
        chunks, metadata_list = process_json_to_chunks(json_data)
        print(f"JSON 条目数: {len(json_data)}")
        print(f"有效文档片段数: {len(chunks)}")
        
    else:
        print("文件类型: 文本文件")
        print(f"分块方法: {args.chunk_method}")
        
        # 处理文本文件
        chunks = split_into_chunks(args.input)
        print(f"已切分片段数: {len(chunks)}")

    if len(chunks) == 0:
        print("❌ 警告: 未找到任何有效文档片段")
        return

    # 显示前几个片段的预览
    print("\n===== 文档片段预览 =====")
    for i, chunk in enumerate(chunks[:3]):
        print(f"[{i+1}] {chunk[:150]}{'...' if len(chunk) > 150 else ''}\n")

    # 2) 向量化
    print("🔄 正在生成向量嵌入...")
    embedding_start_time = time.time()
    embeddings = [embed_chunk(chunk) for chunk in chunks]
    embedding_time = time.time() - embedding_start_time
    
    print(f"✅ 向量化完成，耗时: {embedding_time:.2f}秒")
    print(f"已生成向量数: {len(embeddings)}")
    print(f"向量维度: {len(embeddings[0]) if embeddings else 0}")

    # 3) 存入向量库
    print("💾 正在保存到向量数据库...")
    save_start_time = time.time()
    save_embeddings(chunks, embeddings, args.collection, metadata_list)
    save_time = time.time() - save_start_time
    
    total_time = time.time() - start_time
    
    print(f"✅ 保存完成，耗时: {save_time:.2f}秒")
    print("\n" + "="*50)
    print("📊 构建统计")
    print("="*50)
    print(f"📄 文档片段: {len(chunks)} 个")
    print(f"🔢 向量维度: {len(embeddings[0]) if embeddings else 0}")
    print(f"💾 存储位置: ./chroma_db")
    print(f"📂 集合名称: {args.collection}")
    print(f"⏱️  总耗时: {total_time:.2f}秒")
    if metadata_list:
        print(f"🏷️  元数据: 已保存")
    print("="*50)


if __name__ == "__main__":
    main() 