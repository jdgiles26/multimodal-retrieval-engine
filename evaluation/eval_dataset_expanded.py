"""
扩充版评测数据集 - 包含更多测试查询
为 Duckyo 多模态检索引擎创建更丰富的评测数据集
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import json
from pathlib import Path
from datetime import datetime


@dataclass
class QueryRelevance:
    """查询-文档相关性标注"""
    query: str
    query_image: Optional[str]
    relevant_doc_ids: List[str]
    relevance_scores: Dict[str, float]
    query_type: str
    difficulty: str
    category: str


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


class ExpandedEvalDatasetBuilder:
    """
    扩充版评测数据集构建器
    基于 multimodal_data.json 中的所有字段生成更丰富的测试查询
    """
    
    def __init__(self, data_path: str = "multimodal_data.json"):
        self.data_path = Path(data_path)
        with open(data_path, 'r', encoding='utf-8') as f:
            self.memories = json.load(f)
        
        self.doc_id_to_memory = {m["id"]: m for m in self.memories}
    
    def add_queries(self, queries: List[QueryRelevance]):
        """添加查询到列表"""
        self.queries.extend(queries)
    
    def build_all_queries(self) -> List[QueryRelevance]:
        """构建所有类型的查询"""
        queries = []
        
        # ==================== 1. 城市查询 (Easy) ====================
        city_queries = [
            ("Tokyo", ["tokyo_photo_001", "tokyo_002"]),
            ("Sydney", ["sydney_001", "sydney_002"]),
            ("Paris", ["paris_001"]),
            ("Reykjavik", ["iceland_001"]),
            ("New York", ["nyc_001"]),
            ("日本东京", ["tokyo_photo_001", "tokyo_002"]),
            ("澳大利亚悉尼", ["sydney_001", "sydney_002"]),
            ("法国巴黎", ["paris_001"]),
            ("冰岛雷克雅未克", ["iceland_001"]),
            ("美国纽约", ["nyc_001"]),
        ]
        
        for query, doc_ids in city_queries:
            valid_ids = [d for d in doc_ids if d in self.doc_id_to_memory]
            if valid_ids:
                queries.append(QueryRelevance(
                    query=query,
                    query_image=None,
                    relevant_doc_ids=valid_ids,
                    relevance_scores={doc_id: 1.0 for doc_id in valid_ids},
                    query_type="text",
                    difficulty="easy",
                    category="city"
                ))
        
        # ==================== 2. 地标查询 (Medium) ====================
        landmark_queries = [
            ("东京塔", ["tokyo_photo_001"]),
            ("悉尼歌剧院", ["sydney_001"]),
            ("埃菲尔铁塔", ["paris_001"]),
            ("自由女神像", ["nyc_001"]),
            ("北极光", ["iceland_001"]),
            ("海港大桥", ["sydney_001"]),
            ("浅草寺", ["tokyo_002"]),
            ("极光", ["iceland_001"]),
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
        
        # ==================== 3. 标签查询 (Medium) ====================
        # 从 multimodal_data.json 提取所有 tags
        tag_queries = [
            ("汉服", ["tokyo_photo_001", "paris_001"]),
            ("夜景", ["tokyo_photo_001"]),
            ("海滩", ["sydney_002"]),
            ("冲浪", ["sydney_002"]),
            ("寺庙", ["tokyo_002"]),
            ("和服", ["tokyo_002"]),
            ("极光", ["iceland_001"]),
            ("地标", ["tokyo_photo_001", "sydney_001", "paris_001", "nyc_001"]),
            ("浪漫", ["tokyo_photo_001", "paris_001"]),
            ("自拍", ["tokyo_photo_001"]),
            ("阳光", ["sydney_002"]),
            ("自然奇观", ["iceland_001"]),
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
        
        # ==================== 4. 心情/氛围查询 (Medium-Hard) ====================
        mood_queries = [
            ("浪漫", ["tokyo_photo_001", "paris_001"]),
            ("放松", ["sydney_002"]),
            ("震撼", ["iceland_001"]),
            ("惊艳", ["sydney_001"]),
            ("好奇", ["tokyo_002"]),
            ("庄严", ["nyc_001"]),
            ("浪漫的旅行", ["tokyo_photo_001", "paris_001"]),
            ("放松的海滩时光", ["sydney_002"]),
            ("震撼的自然景观", ["iceland_001"]),
            ("美丽的海港", ["sydney_001"]),
        ]
        
        for query, doc_ids in mood_queries:
            valid_ids = [d for d in doc_ids if d in self.doc_id_to_memory]
            if valid_ids:
                difficulty = "medium" if len(valid_ids) == 1 else "hard"
                queries.append(QueryRelevance(
                    query=query,
                    query_image=None,
                    relevant_doc_ids=valid_ids,
                    relevance_scores={doc_id: 1.0 for doc_id in valid_ids},
                    query_type="text",
                    difficulty=difficulty,
                    category="mood"
                ))
        
        # ==================== 5. 穿着/服装查询 (Medium) ====================
        outfit_queries = [
            ("汉服", ["tokyo_photo_001", "paris_001"]),
            ("和服", ["tokyo_002"]),
            ("休闲", ["sydney_001", "nyc_001"]),
            ("泳装", ["sydney_002"]),
            ("保暖", ["iceland_001"]),
            ("穿传统服装", ["tokyo_photo_001", "paris_001", "tokyo_002"]),
        ]
        
        for query, doc_ids in outfit_queries:
            valid_ids = [d for d in doc_ids if d in self.doc_id_to_memory]
            if valid_ids:
                queries.append(QueryRelevance(
                    query=query,
                    query_image=None,
                    relevant_doc_ids=valid_ids,
                    relevance_scores={doc_id: 1.0 for doc_id in valid_ids},
                    query_type="text",
                    difficulty="medium",
                    category="outfit"
                ))
        
        # ==================== 6. 语义理解查询 (Hard) ====================
        semantic_queries = [
            ("穿传统服装的旅行", ["tokyo_photo_001", "paris_001", "tokyo_002"]),
            ("著名的地标建筑", ["tokyo_photo_001", "sydney_001", "paris_001", "nyc_001"]),
            ("海边或海滩", ["sydney_002"]),
            ("夜晚的城市风景", ["tokyo_photo_001"]),
            ("中西合璧的旅行照片", ["tokyo_photo_001", "paris_001"]),
            ("标志性的城市景点", ["tokyo_photo_001", "sydney_001", "paris_001", "nyc_001"]),
            ("户外活动", ["sydney_002"]),
            ("文化体验", ["tokyo_002"]),
            ("自然与城市结合", ["sydney_001"]),
            ("历史与现代", ["tokyo_002", "nyc_001"]),
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
        
        # ==================== 7. 图片搜索查询 (Image) ====================
        image_queries = [
            ("tokyo_photo_001", "tokyo_photo_001"),
            ("sydney_001", "sydney_001"),
            ("sydney_002", "sydney_002"),
            ("paris_001", "paris_001"),
            ("iceland_001", "iceland_001"),
            ("nyc_001", "nyc_001"),
            ("tokyo_002", "tokyo_002"),
        ]
        
        for query_id, image_id in image_queries:
            memory = self.doc_id_to_memory.get(query_id)
            if memory and memory.get("image_url"):
                queries.append(QueryRelevance(
                    query="",
                    query_image=memory["image_url"],
                    relevant_doc_ids=[image_id],
                    relevance_scores={image_id: 1.0},
                    query_type="image",
                    difficulty="medium",
                    category="image_to_image"
                ))
        
        # ==================== 8. 混合搜索查询 (Hybrid) ====================
        hybrid_queries = [
            ("东京夜景", "tokyo_photo_001", ["tokyo_photo_001"]),
            ("悉尼海港", "sydney_001", ["sydney_001"]),
            ("巴黎浪漫", "paris_001", ["paris_001"]),
            ("冰岛极光", "iceland_001", ["iceland_001"]),
            ("纽约地标", "nyc_001", ["nyc_001"]),
            ("东京和服", "tokyo_002", ["tokyo_002"]),
            ("悉尼海滩", "sydney_002", ["sydney_002"]),
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
    
    def build_dataset(self) -> EvalDataset:
        """构建完整评测数据集"""
        all_queries = self.build_all_queries()
        
        return EvalDataset(
            name="Duckyo Travel Memories Eval (Expanded)",
            description="Duckyo 多模态旅行记忆检索引擎扩充版评测数据集",
            queries=all_queries,
            corpus_size=len(self.memories),
            created_at=datetime.now().isoformat()
        )
    
    def save_dataset(self, output_path: str = "evaluation/eval_dataset_expanded.json"):
        """保存评测数据集"""
        dataset = self.build_dataset()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset.to_dict(), f, ensure_ascii=False, indent=2)
        
        print(f"✅ 扩充版评测数据集已保存: {output_path}")
        print(f"   - 查询总数: {len(dataset.queries)}")
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
        for t, c in sorted(type_counts.items()):
            print(f"   - {t}: {c}")
        
        print(f"\n📊 难度分布:")
        for d, c in sorted(difficulty_counts.items()):
            print(f"   - {d}: {c}")
        
        print(f"\n📊 类别分布:")
        for c, cnt in sorted(category_counts.items()):
            print(f"   - {c}: {cnt}")
        
        return dataset


if __name__ == "__main__":
    builder = ExpandedEvalDatasetBuilder("multimodal_data.json")
    builder.save_dataset("evaluation/eval_dataset_expanded.json")