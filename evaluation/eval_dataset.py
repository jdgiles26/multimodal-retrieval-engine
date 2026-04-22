"""
评测数据集构建与管理
为 Duckyo 多模态检索引擎创建标准评测数据集
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import json
import random
from pathlib import Path


@dataclass
class QueryRelevance:
    """查询-文档相关性标注"""
    query: str  # 查询文本
    query_image: Optional[str]  # 查询图片（可选）
    relevant_doc_ids: List[str]  # 相关文档ID列表
    relevance_scores: Dict[str, float]  # 文档ID -> 相关性分数 (0-1)
    query_type: str  # 查询类型: text, image, hybrid
    difficulty: str  # 难度: easy, medium, hard
    category: str  # 类别: city, landmark, mood, outfit, tag


@dataclass
class EvalDataset:
    """评测数据集"""
    name: str
    description: str
    queries: List[QueryRelevance]
    corpus_size: int
    created_at: str
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "corpus_size": self.corpus_size,
            "created_at": self.created_at,
            "queries": [
                {
                    "query": q.query,
                    "query_image": q.query_image,
                    "relevant_doc_ids": q.relevant_doc_ids,
                    "relevance_scores": q.relevance_scores,
                    "query_type": q.query_type,
                    "difficulty": q.difficulty,
                    "category": q.category
                }
                for q in self.queries
            ]
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "EvalDataset":
        queries = [
            QueryRelevance(
                query=q["query"],
                query_image=q.get("query_image"),
                relevant_doc_ids=q["relevant_doc_ids"],
                relevance_scores=q["relevance_scores"],
                query_type=q["query_type"],
                difficulty=q["difficulty"],
                category=q["category"]
            )
            for q in data["queries"]
        ]
        return cls(
            name=data["name"],
            description=data["description"],
            queries=queries,
            corpus_size=data["corpus_size"],
            created_at=data["created_at"]
        )


class DuckyoEvalDatasetBuilder:
    """
    Duckyo 旅行记忆评测数据集构建器
    基于现有的 multimodal_data.json 构建评测集
    """
    
    def __init__(self, data_path: str = "multimodal_data.json"):
        self.data_path = Path(data_path)
        with open(data_path, 'r', encoding='utf-8') as f:
            self.memories = json.load(f)
        
        self.doc_id_to_memory = {m["id"]: m for m in self.memories}
        self.city_to_docs = self._group_by_city()
        self.tag_to_docs = self._group_by_tag()
        self.mood_to_docs = self._group_by_mood()
        
    def _group_by_city(self) -> Dict[str, List[str]]:
        """按城市分组"""
        groups = {}
        for m in self.memories:
            city = m.get("city", "")
            if city:
                groups.setdefault(city, []).append(m["id"])
        return groups
    
    def _group_by_tag(self) -> Dict[str, List[str]]:
        """按标签分组"""
        groups = {}
        for m in self.memories:
            for tag in m.get("tags", []):
                groups.setdefault(tag, []).append(m["id"])
        return groups
    
    def _group_by_mood(self) -> Dict[str, List[str]]:
        """按心情分组"""
        groups = {}
        for m in self.memories:
            mood = m.get("mood", "")
            if mood:
                groups.setdefault(mood, []).append(m["id"])
        return groups
    
    def build_text_search_queries(self) -> List[QueryRelevance]:
        """构建文本搜索评测查询"""
        queries = []
        
        # 城市搜索 (Easy)
        for city, doc_ids in self.city_to_docs.items():
            queries.append(QueryRelevance(
                query=f"{city}的照片",
                query_image=None,
                relevant_doc_ids=doc_ids,
                relevance_scores={doc_id: 1.0 for doc_id in doc_ids},
                query_type="text",
                difficulty="easy",
                category="city"
            ))
        
        # 地标搜索 (Medium)
        landmark_queries = [
            ("东京塔", ["tokyo_photo_001"]),
            ("悉尼歌剧院", ["sydney_001"]),
            ("埃菲尔铁塔", ["paris_001"]),
            ("自由女神像", ["nyc_001"]),
            ("北极光", ["iceland_001"]),
        ]
        for query, doc_ids in landmark_queries:
            valid_ids = [d for d in doc_ids if d in self.doc_id_to_memory]
            if valid_ids:
                queries.append(QueryRelevance(
                    query=query,
                    query_image=None,
                    relevant_doc_ids=valid_ids,
                    relevance_scores={doc_id: 1.0 for doc_id in valid_ids},
                    query_type="text",
                    difficulty="medium",
                    category="landmark"
                ))
        
        # 标签搜索 (Medium)
        tag_queries = [
            ("汉服", ["tokyo_photo_001", "paris_001"]),
            ("海滩", ["sydney_002"]),
            ("夜景", ["tokyo_photo_001"]),
        ]
        for query, doc_ids in tag_queries:
            valid_ids = [d for d in doc_ids if d in self.doc_id_to_memory]
            if valid_ids:
                queries.append(QueryRelevance(
                    query=query,
                    query_image=None,
                    relevant_doc_ids=valid_ids,
                    relevance_scores={doc_id: 1.0 for doc_id in valid_ids},
                    query_type="text",
                    difficulty="medium",
                    category="tag"
                ))
        
        # 心情/氛围搜索 (Hard)
        mood_queries = [
            ("浪漫的旅行", ["tokyo_photo_001", "paris_001"]),
            ("放松的海滩时光", ["sydney_002"]),
            ("震撼的自然景观", ["iceland_001"]),
        ]
        for query, doc_ids in mood_queries:
            valid_ids = [d for d in doc_ids if d in self.doc_id_to_memory]
            if valid_ids:
                queries.append(QueryRelevance(
                    query=query,
                    query_image=None,
                    relevant_doc_ids=valid_ids,
                    relevance_scores={doc_id: 1.0 for doc_id in valid_ids},
                    query_type="text",
                    difficulty="hard",
                    category="mood"
                ))
        
        # 语义搜索 (Hard)
        semantic_queries = [
            ("穿传统服装的旅行", ["tokyo_photo_001", "paris_001"]),
            ("著名的地标建筑", ["tokyo_photo_001", "sydney_001", "paris_001", "nyc_001"]),
            ("海边或海滩", ["sydney_002"]),
        ]
        for query, doc_ids in semantic_queries:
            valid_ids = [d for d in doc_ids if d in self.doc_id_to_memory]
            if valid_ids:
                queries.append(QueryRelevance(
                    query=query,
                    query_image=None,
                    relevant_doc_ids=valid_ids,
                    relevance_scores={doc_id: 1.0 for doc_id in valid_ids},
                    query_type="text",
                    difficulty="hard",
                    category="semantic"
                ))
        
        return queries
    
    def build_image_search_queries(self) -> List[QueryRelevance]:
        """构建图搜图评测查询 (需要实际图片URL)"""
        queries = []
        
        # 基于已有图片的图搜图查询
        image_queries = [
            ("tokyo_photo_001", "tokyo_photo_001", ["tokyo_photo_001"]),
            ("sydney_001", "sydney_001", ["sydney_001"]),
            ("paris_001", "paris_001", ["paris_001"]),
        ]
        
        for query_id, image_id, expected_ids in image_queries:
            memory = self.doc_id_to_memory.get(query_id)
            if memory and memory.get("image_url"):
                valid_ids = [d for d in expected_ids if d in self.doc_id_to_memory]
                if valid_ids:
                    queries.append(QueryRelevance(
                        query="",
                        query_image=memory["image_url"],
                        relevant_doc_ids=valid_ids,
                        relevance_scores={doc_id: 1.0 for doc_id in valid_ids},
                        query_type="image",
                        difficulty="medium",
                        category="image_to_image"
                    ))
        
        return queries
    
    def build_hybrid_search_queries(self) -> List[QueryRelevance]:
        """构建图文混合搜索评测查询"""
        queries = []
        
        # 图文混合查询
        hybrid_queries = [
            ("东京夜景", "tokyo_photo_001", ["tokyo_photo_001"]),
            ("悉尼海港", "sydney_001", ["sydney_001"]),
        ]
        
        for text_query, image_id, expected_ids in hybrid_queries:
            memory = self.doc_id_to_memory.get(image_id)
            if memory and memory.get("image_url"):
                valid_ids = [d for d in expected_ids if d in self.doc_id_to_memory]
                if valid_ids:
                    queries.append(QueryRelevance(
                        query=text_query,
                        query_image=memory["image_url"],
                        relevant_doc_ids=valid_ids,
                        relevance_scores={doc_id: 1.0 for doc_id in valid_ids},
                        query_type="hybrid",
                        difficulty="hard",
                        category="hybrid"
                    ))
        
        return queries
    
    def build_full_dataset(self) -> EvalDataset:
        """构建完整评测数据集"""
        from datetime import datetime
        
        text_queries = self.build_text_search_queries()
        image_queries = self.build_image_search_queries()
        hybrid_queries = self.build_hybrid_search_queries()
        
        all_queries = text_queries + image_queries + hybrid_queries
        
        return EvalDataset(
            name="Duckyo Travel Memories Eval",
            description="Duckyo 多模态旅行记忆检索引擎评测数据集",
            queries=all_queries,
            corpus_size=len(self.memories),
            created_at=datetime.now().isoformat()
        )
    
    def save_dataset(self, output_path: str = "evaluation/eval_dataset.json"):
        """保存评测数据集到文件"""
        dataset = self.build_full_dataset()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset.to_dict(), f, ensure_ascii=False, indent=2)
        
        print(f"✅ 评测数据集已保存: {output_path}")
        print(f"   - 查询数量: {len(dataset.queries)}")
        print(f"   - 语料库大小: {dataset.corpus_size}")
        
        # 统计
        type_counts = {}
        difficulty_counts = {}
        category_counts = {}
        
        for q in dataset.queries:
            type_counts[q.query_type] = type_counts.get(q.query_type, 0) + 1
            difficulty_counts[q.difficulty] = difficulty_counts.get(q.difficulty, 0) + 1
            category_counts[q.category] = category_counts.get(q.category, 0) + 1
        
        print(f"\n📊 查询类型分布:")
        for t, c in type_counts.items():
            print(f"   - {t}: {c}")
        
        print(f"\n📊 难度分布:")
        for d, c in difficulty_counts.items():
            print(f"   - {d}: {c}")
        
        return dataset


if __name__ == "__main__":
    # 构建并保存评测数据集
    builder = DuckyoEvalDatasetBuilder("multimodal_data.json")
    builder.save_dataset("evaluation/eval_dataset.json")
