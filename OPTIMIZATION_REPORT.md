# Multimodal Retrieval Engine Optimization Report

## Current Status Overview

### Data Scale
- **Total records**: 4 (critically insufficient)
- **Media type distribution**:
  - Image: 2
  - Video: 1
  - Test: 1 (test data)
- **City coverage**: Sydney, Paris, Tokyo, test

### Performance Metrics (based on 76 query evaluation)
| Metric | Value | Assessment |
|--------|-------|------------|
| Recall@5 | 28.9% | Low |
| Recall@10 | 40.5% | Low |
| Precision@5 | 7.4% | Very low |
| NDCG@5 | 22.8% | Low |
| MRR | 23.9% | Low |
| MAP | 21.7% | Low |
| P50 Latency | 3.3s | Slow |
| P95 Latency | 5.7s | Very slow |

---

## Critical Issues

### 1. Insufficient Data Volume
**Problem**: Only 4 records, unable to support effective multimodal retrieval
**Impact**: Almost no queries can recall relevant results
**Solution**: Import large volumes of real travel data

### 2. Frequent Rerank API Failures
**Error**: `Input should be a valid string: input.query & input.documents.0`
**Root cause**:
- Query text contains special characters (emoji)
- Document format does not meet qwen3-rerank API requirements
**Solution**: Sanitize input text, escape special characters

### 3. Milvus Query Expression Errors
**Error**: `cannot parse expression: text like "Shibuya night view..."`
**Root cause**: Text contains quotes and special characters, causing SQL-like query failures
**Solution**: Use parameterized queries or escape special characters

### 4. High Latency
**P95 latency 5.7 seconds**, poor user experience
**Root causes**:
- Multimodal embedding API call overhead
- Rerank step adds extra latency
- Network round-trip time

### 5. High Empty Field Rate
| Field | Empty Rate |
|-------|------------|
| video_url | 75% |
| outfit | 75% |
| image_url | 25% |
| tags | 25% |
| mood | 25% |

---

## Optimization Opportunities

### 1. Data Quality Improvement
- **Standardize city names**: Keep English consistent (Tokyo, Sydney, Paris)
- **Fill missing fields**: Supplement tags, mood, outfit metadata
- **Remove test data**: Delete `test_record`
- **Data validation**: Check required fields before ingestion

### 2. Recall Rate Optimization
- **Increase data volume**: At least 100+ records for effective retrieval
- **Data augmentation**: Generate multi-perspective descriptions per record
- **Synonym expansion**: "Tokyo" <-> "Tokyo" <-> "capital of Japan"

### 3. Latency Optimization
- **Async processing**: Execute embedding and rerank in parallel
- **Caching**: Cache embeddings for popular queries
- **Connection pooling**: Reuse Milvus and DashScope connections
- **Early return**: Skip rerank if coarse filter returns few results

### 4. Video Retrieval Enhancement
- **Video preview thumbnails**: Generate first-frame thumbnails for videos
- **Video segmentation**: Split long videos and encode segments
- **Video metadata**: Extract duration, resolution, and other info

### 5. Image-to-Image Optimization
- **Local image support**: Improve base64 conversion logic
- **Image preprocessing**: Standardize image sizes and formats
- **Similarity threshold**: Add filtering logic for low-similarity results

---

## Recommended Optimization Plan

### Short-term (1-2 days)

1. **Fix Rerank API errors**
   ```python
   import re
   clean_query = re.sub(r'[^\w\s]', '', query)
   ```

2. **Fix Milvus query errors**
   ```python
   safe_text = text.replace('"', '\\"').replace("'", "\\'")
   ```

3. **Delete test data**
   ```python
   collection.delete('doc_id == "test_record"')
   ```

4. **Standardize city names**

### Medium-term (1 week)

1. **Batch import real data**
   - Import from social media notes
   - Import from travel photo libraries
   - Target: 100+ records

2. **Implement query caching**
   ```python
   from functools import lru_cache
   
   @lru_cache(maxsize=1000)
   def get_embedding(text):
       return embedding_api(text)
   ```

3. **Add data validation**
   ```python
   def validate_memory(mem):
       required = ['id', 'content', 'city']
       for field in required:
           if not mem.get(field):
               raise ValueError(f"Missing {field}")
   ```

### Long-term (1 month)

1. **Implement hybrid index**
   - Vector index + inverted index
   - Support keyword filtering

2. **Multimodal fusion strategy optimization**
   - Dynamic text weight vs image weight adjustment
   - Video keyframe extraction

3. **A/B testing framework**
   - Compare different embedding models
   - Compare different rerank strategies

---

## Expected Improvements

| Optimization | Expected Gain |
|--------------|---------------|
| Data volume to 100+ | Recall@5: 28% -> 60%+ |
| Fix Rerank errors | Success rate: current failure rate -> 100% |
| Latency optimization | P95: 5.7s -> 2s |
| Query caching | Hot queries: 3s -> 0.1s |

---

## Priority Ranking

1. **P0 (Urgent)**: Fix Rerank and Milvus query errors
2. **P1 (Important)**: Batch import real data (100+ records)
3. **P2 (Recommended)**: Implement query caching
4. **P3 (Optional)**: Data validation and standardization

---

*Report generated: 2026-04-20*
*Engine version: LlamaIndex + Qwen3-VL + Milvus*
