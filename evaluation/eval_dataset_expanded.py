"""
Expanded evaluation dataset - contains more test queries
Create a richer evaluation dataset for Duckyo multimodal retrieval engine
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import json
from pathlib import Path
from datetime import datetime


@dataclass
class QueryRelevance:
    """Query-document relevance annotation"""
    query: str
    query_image: Optional[str]
    relevant_doc_ids: List[str]
    relevance_scores: Dict[str, float]
    query_type: str
    difficulty: str
    category: str


@dataclass
class EvalDataset:
    """Evaluation Dataset"""
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
    Expanded evaluation dataset builder
    Generate richer test queries based on all fields in multimodal_data.json
    """
    
    def __init__(self, data_path: str = "multimodal_data.json"):
        self.data_path = Path(data_path)
        with open(data_path, 'r', encoding='utf-8') as f:
            self.memories = json.load(f)
        
        self.doc_id_to_memory = {m["id"]: m for m in self.memories}
    
    def add_queries(self, queries: List[QueryRelevance]):
        """Add queries to list"""
        self.queries.extend(queries)
    
    def build_all_queries(self) -> List[QueryRelevance]:
        """Build all types of queries"""
        queries = []
        
        # ==================== 1. City queries (Easy) ====================
        city_queries = [
            ("Tokyo", ["tokyo_photo_001", "tokyo_002"]),
            ("Sydney", ["sydney_001", "sydney_002"]),
            ("Paris", ["paris_001"]),
            ("Reykjavik", ["iceland_001"]),
            ("New York", ["nyc_001"]),
            ("Tokyo, Japan", ["tokyo_photo_001", "tokyo_002"]),
            ("Sydney, Australia", ["sydney_001", "sydney_002"]),
            ("Paris, France", ["paris_001"]),
            ("Reykjavik, Iceland", ["iceland_001"]),
            ("New York, USA", ["nyc_001"]),
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
        
        # ==================== 2. Landmark queries (Medium) ====================
        landmark_queries = [
            ("Tokyo Tower", ["tokyo_photo_001"]),
            ("Sydney Opera House", ["sydney_001"]),
            ("Eiffel Tower", ["paris_001"]),
            ("Statue of Liberty", ["nyc_001"]),
            ("Northern Lights", ["iceland_001"]),
            ("Harbour Bridge", ["sydney_001"]),
            ("Sensoji Temple", ["tokyo_002"]),
            ("Aurora", ["iceland_001"]),
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
        
        # ==================== 3. Tag queries (Medium) ====================
        # Extract all tags from multimodal_data.json
        tag_queries = [
            ("Hanfu", ["tokyo_photo_001", "paris_001"]),
            ("night view", ["tokyo_photo_001"]),
            ("beach", ["sydney_002"]),
            ("surfing", ["sydney_002"]),
            ("temple", ["tokyo_002"]),
            ("Kimono", ["tokyo_002"]),
            ("Aurora", ["iceland_001"]),
            ("landmark", ["tokyo_photo_001", "sydney_001", "paris_001", "nyc_001"]),
            ("romantic", ["tokyo_photo_001", "paris_001"]),
            ("selfie", ["tokyo_photo_001"]),
            ("sunshine", ["sydney_002"]),
            ("natural wonders", ["iceland_001"]),
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
        
        # ==================== 4. Mood/atmosphere queries (Medium-Hard) ====================
        mood_queries = [
            ("romantic", ["tokyo_photo_001", "paris_001"]),
            ("relaxing", ["sydney_002"]),
            ("breathtaking", ["iceland_001"]),
            ("stunning", ["sydney_001"]),
            ("curious", ["tokyo_002"]),
            ("solemn", ["nyc_001"]),
            ("romantic travel", ["tokyo_photo_001", "paris_001"]),
            ("relaxing beach time", ["sydney_002"]),
            ("breathtaking natural scenery", ["iceland_001"]),
            ("beautiful harbor", ["sydney_001"]),
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
        
        # ==================== 5. Clothing/outfit queries (Medium) ====================
        outfit_queries = [
            ("Hanfu", ["tokyo_photo_001", "paris_001"]),
            ("Kimono", ["tokyo_002"]),
            ("casual", ["sydney_001", "nyc_001"]),
            ("swimwear", ["sydney_002"]),
            ("warm clothing", ["iceland_001"]),
            ("wearing traditional clothing", ["tokyo_photo_001", "paris_001", "tokyo_002"]),
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
        
        # ==================== 6. Semantic understanding queries (Hard) ====================
        semantic_queries = [
            ("travel in traditional clothing", ["tokyo_photo_001", "paris_001", "tokyo_002"]),
            ("famous landmark buildings", ["tokyo_photo_001", "sydney_001", "paris_001", "nyc_001"]),
            ("seaside or beach", ["sydney_002"]),
            ("night city scenery", ["tokyo_photo_001"]),
            ("travel photos blending East and West", ["tokyo_photo_001", "paris_001"]),
            ("iconic city attractions", ["tokyo_photo_001", "sydney_001", "paris_001", "nyc_001"]),
            ("outdoor activities", ["sydney_002"]),
            ("cultural experience", ["tokyo_002"]),
            ("nature and city combined", ["sydney_001"]),
            ("history and modernity", ["tokyo_002", "nyc_001"]),
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
        
        # ==================== 7. Image search queries (Image) ====================
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
        
        # ==================== 8. Hybrid search queries (Hybrid) ====================
        hybrid_queries = [
            ("Tokyo night view", "tokyo_photo_001", ["tokyo_photo_001"]),
            ("Sydney harbor", "sydney_001", ["sydney_001"]),
            ("Paris romance", "paris_001", ["paris_001"]),
            ("Iceland aurora", "iceland_001", ["iceland_001"]),
            ("New York landmarks", "nyc_001", ["nyc_001"]),
            ("Tokyo Kimono", "tokyo_002", ["tokyo_002"]),
            ("Sydney beach", "sydney_002", ["sydney_002"]),
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
        """Build full evaluation dataset"""
        all_queries = self.build_all_queries()
        
        return EvalDataset(
            name="Duckyo Travel Memories Eval (Expanded)",
            description="Duckyo Multimodal Travel Memory Retrieval Engine Expanded Evaluation Dataset",
            queries=all_queries,
            corpus_size=len(self.memories),
            created_at=datetime.now().isoformat()
        )
    
    def save_dataset(self, output_path: str = "evaluation/eval_dataset_expanded.json"):
        """Save evaluation dataset"""
        dataset = self.build_dataset()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset.to_dict(), f, ensure_ascii=False, indent=2)
        
        print(f" Expanded evaluation dataset saved: {output_path}")
        print(f"   - Total queries: {len(dataset.queries)}")
        print(f"   - Corpus size: {dataset.corpus_size}")
        
        # Statistics
        type_counts = {}
        difficulty_counts = {}
        category_counts = {}
        
        for q in dataset.queries:
            type_counts[q.query_type] = type_counts.get(q.query_type, 0) + 1
            difficulty_counts[q.difficulty] = difficulty_counts.get(q.difficulty, 0) + 1
            category_counts[q.category] = category_counts.get(q.category, 0) + 1
        
        print(f"\n Query type distribution:")
        for t, c in sorted(type_counts.items()):
            print(f"   - {t}: {c}")
        
        print(f"\n Difficulty distribution:")
        for d, c in sorted(difficulty_counts.items()):
            print(f"   - {d}: {c}")
        
        print(f"\n Category distribution:")
        for c, cnt in sorted(category_counts.items()):
            print(f"   - {c}: {cnt}")
        
        return dataset


if __name__ == "__main__":
    builder = ExpandedEvalDatasetBuilder("multimodal_data.json")
    builder.save_dataset("evaluation/eval_dataset_expanded.json")