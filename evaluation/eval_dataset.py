"""
Evaluation dataset construction and management
Create standard evaluation dataset for Duckyo multimodal retrieval engine
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import json
import random
from pathlib import Path


@dataclass
class QueryRelevance:
    """Query-document relevance annotation"""
    query: str  # Query text
    query_image: Optional[str]  # Query image (optional)
    relevant_doc_ids: List[str]  # Relevant document ID list
    relevance_scores: Dict[str, float]  # Document ID -> relevance score (0-1)
    query_type: str  # Query type: text, image, hybrid
    difficulty: str  # Difficulty: easy, medium, hard
    category: str  # Category: city, landmark, mood, outfit, tag


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
    Duckyo Travel Memory Evaluation Dataset Builder
    Build evaluation set based on existing multimodal_data.json
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
        """Group by city"""
        groups = {}
        for m in self.memories:
            city = m.get("city", "")
            if city:
                groups.setdefault(city, []).append(m["id"])
        return groups
    
    def _group_by_tag(self) -> Dict[str, List[str]]:
        """Group by tag"""
        groups = {}
        for m in self.memories:
            for tag in m.get("tags", []):
                groups.setdefault(tag, []).append(m["id"])
        return groups
    
    def _group_by_mood(self) -> Dict[str, List[str]]:
        """Group by mood"""
        groups = {}
        for m in self.memories:
            mood = m.get("mood", "")
            if mood:
                groups.setdefault(mood, []).append(m["id"])
        return groups
    
    def build_text_search_queries(self) -> List[QueryRelevance]:
        """Build text search evaluation queries"""
        queries = []
        
        # City search (Easy)
        for city, doc_ids in self.city_to_docs.items():
            queries.append(QueryRelevance(
                query=f"{city} photos",
                query_image=None,
                relevant_doc_ids=doc_ids,
                relevance_scores={doc_id: 1.0 for doc_id in doc_ids},
                query_type="text",
                difficulty="easy",
                category="city"
            ))
        
        # Landmark search (Medium)
        landmark_queries = [
            ("Tokyo Tower", ["tokyo_photo_001"]),
            ("Sydney Opera House", ["sydney_001"]),
            ("Eiffel Tower", ["paris_001"]),
            ("Statue of Liberty", ["nyc_001"]),
            ("Northern Lights", ["iceland_001"]),
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
        
        # Tag search (Medium)
        tag_queries = [
            ("Hanfu", ["tokyo_photo_001", "paris_001"]),
            ("beach", ["sydney_002"]),
            ("night view", ["tokyo_photo_001"]),
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
        
        # Mood/atmosphere search (Hard)
        mood_queries = [
            ("romantic travel", ["tokyo_photo_001", "paris_001"]),
            ("relaxing beach time", ["sydney_002"]),
            ("breathtaking natural scenery", ["iceland_001"]),
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
        
        # Semantic search (Hard)
        semantic_queries = [
            ("travel in traditional clothing", ["tokyo_photo_001", "paris_001"]),
            ("famous landmark buildings", ["tokyo_photo_001", "sydney_001", "paris_001", "nyc_001"]),
            ("seaside or beach", ["sydney_002"]),
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
        """Build image-to-image search evaluation queries (requires actual image URLs)"""
        queries = []
        
        # Image-to-image queries based on existing images
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
        """Build hybrid search evaluation queries"""
        queries = []
        
        # Text-image hybrid queries
        hybrid_queries = [
            ("Tokyo night view", "tokyo_photo_001", ["tokyo_photo_001"]),
            ("Sydney harbor", "sydney_001", ["sydney_001"]),
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
        """Build full evaluation dataset"""
        from datetime import datetime
        
        text_queries = self.build_text_search_queries()
        image_queries = self.build_image_search_queries()
        hybrid_queries = self.build_hybrid_search_queries()
        
        all_queries = text_queries + image_queries + hybrid_queries
        
        return EvalDataset(
            name="Duckyo Travel Memories Eval",
            description="Duckyo Multimodal Travel Memory Retrieval Engine Evaluation Dataset",
            queries=all_queries,
            corpus_size=len(self.memories),
            created_at=datetime.now().isoformat()
        )
    
    def save_dataset(self, output_path: str = "evaluation/eval_dataset.json"):
        """Save evaluation dataset to file"""
        dataset = self.build_full_dataset()
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset.to_dict(), f, ensure_ascii=False, indent=2)
        
        print(f" Evaluation dataset saved: {output_path}")
        print(f"   - Query count: {len(dataset.queries)}")
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
        for t, c in type_counts.items():
            print(f"   - {t}: {c}")
        
        print(f"\n Difficulty distribution:")
        for d, c in difficulty_counts.items():
            print(f"   - {d}: {c}")
        
        return dataset


if __name__ == "__main__":
    # Build and save evaluation dataset
    builder = DuckyoEvalDatasetBuilder("multimodal_data.json")
    builder.save_dataset("evaluation/eval_dataset.json")
