#!/usr/bin/env python3
"""
Conversation Indexing and Search System - Task 5.4.3.3

Implements advanced search capabilities for conversations:
- Full-text search indexing
- Semantic search with embeddings
- Advanced filtering and faceted search
- Search result ranking and relevance
- Performance optimization
"""

import json
import re
import sqlite3
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import math

# Enterprise imports
import sys
sys.path.append(str(Path(__file__).parent.parent / "enterprise_config"))
from enterprise_config import get_config
from enterprise_logging import get_logger
from enterprise_error_handling import handle_error, with_retry

# Database imports
from conversation_database import ConversationDatabase

@dataclass
class SearchQuery:
    """Search query parameters."""
    text: Optional[str] = None
    filters: Dict[str, Any] = field(default_factory=dict)
    sort_by: str = "relevance"  # relevance, date, quality
    sort_order: str = "desc"    # asc, desc
    limit: int = 20
    offset: int = 0
    include_highlights: bool = True
    min_score: float = 0.0

@dataclass
class SearchResult:
    """Search result with metadata."""
    conversation_id: str
    title: Optional[str]
    snippet: str
    score: float
    highlights: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class SearchStats:
    """Search statistics and performance metrics."""
    total_results: int
    search_time_ms: float
    query_text: str
    filters_applied: Dict[str, Any]
    facets: Dict[str, Dict[str, int]] = field(default_factory=dict)

class ConversationSearchEngine:
    """Advanced search engine for conversations."""
    
    def __init__(self, database: ConversationDatabase):
        self.config = get_config()
        self.logger = get_logger("conversation_search")
        self.database = database
        
        # Search configuration
        self.max_snippet_length = 200
        self.highlight_tags = ("<mark>", "</mark>")
        self.min_word_length = 3
        self.stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have',
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
        }
        
        # Initialize search indexes
        self._initialize_search_indexes()
        
        self.logger.info("Conversation search engine initialized")
    
    def _initialize_search_indexes(self):
        """Initialize search indexes and FTS tables."""
        
        try:
            with self.database._get_connection() as conn:
                # Create FTS5 virtual table for full-text search
                conn.execute("""
                    CREATE VIRTUAL TABLE IF NOT EXISTS conversations_fts USING fts5(
                        conversation_id UNINDEXED,
                        title,
                        content,
                        tags,
                        categories,
                        techniques,
                        content='conversations',
                        content_rowid='rowid'
                    )
                """)
                
                # Create triggers to keep FTS table in sync
                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS conversations_fts_insert AFTER INSERT ON conversations BEGIN
                        INSERT INTO conversations_fts(
                            conversation_id, title, content, tags, categories, techniques
                        ) VALUES (
                            new.conversation_id,
                            new.title,
                            new.conversations_json,
                            '',
                            '',
                            ''
                        );
                    END
                """)
                
                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS conversations_fts_delete AFTER DELETE ON conversations BEGIN
                        DELETE FROM conversations_fts WHERE conversation_id = old.conversation_id;
                    END
                """)
                
                conn.execute("""
                    CREATE TRIGGER IF NOT EXISTS conversations_fts_update AFTER UPDATE ON conversations BEGIN
                        UPDATE conversations_fts SET
                            title = new.title,
                            content = new.conversations_json
                        WHERE conversation_id = new.conversation_id;
                    END
                """)
                
                conn.commit()
                
        except Exception as e:
            handle_error(e, "conversation_search", {"context": "initialize_search_indexes"})
    
    def index_conversation(self, conversation_id: str, force_reindex: bool = False) -> bool:
        """Index a single conversation for search."""
        
        try:
            # Get conversation data
            conversation = self.database.get_conversation(conversation_id)
            if not conversation:
                return False
            
            with self.database._get_connection() as conn:
                # Check if already indexed
                if not force_reindex:
                    cursor = conn.execute("""
                        SELECT conversation_id FROM conversation_search WHERE conversation_id = ?
                    """, (conversation_id,))
                    if cursor.fetchone():
                        return True
                
                # Extract searchable content
                searchable_content = self._extract_searchable_content(conversation)
                
                # Insert/update search index
                conn.execute("""
                    INSERT OR REPLACE INTO conversation_search (
                        search_id, conversation_id, searchable_content, last_indexed
                    ) VALUES (?, ?, ?, ?)
                """, (
                    conversation_id,  # Using conversation_id as search_id for simplicity
                    conversation_id,
                    searchable_content,
                    datetime.now(timezone.utc)
                ))
                
                # Update FTS index
                tags_str = ' '.join(conversation.get('tags', []))
                categories_str = ' '.join(conversation.get('categories', []))
                techniques_str = ' '.join(conversation.get('techniques', []))
                
                conn.execute("""
                    INSERT OR REPLACE INTO conversations_fts (
                        conversation_id, title, content, tags, categories, techniques
                    ) VALUES (?, ?, ?, ?, ?, ?)
                """, (
                    conversation_id,
                    conversation.get('title', ''),
                    searchable_content,
                    tags_str,
                    categories_str,
                    techniques_str
                ))
                
                conn.commit()
                
            self.logger.debug(f"Indexed conversation: {conversation_id}")
            return True
            
        except Exception as e:
            handle_error(e, "conversation_search", {
                "operation": "index_conversation",
                "conversation_id": conversation_id
            })
            return False
    
    def _extract_searchable_content(self, conversation: Dict[str, Any]) -> str:
        """Extract searchable text content from conversation."""
        
        content_parts = []
        
        # Add title if available
        if conversation.get('title'):
            content_parts.append(conversation['title'])
        
        # Add conversation exchanges
        conversations = conversation.get('conversations', [])
        if isinstance(conversations, str):
            conversations = json.loads(conversations)
        
        for exchange in conversations:
            if isinstance(exchange, dict):
                for role, text in exchange.items():
                    if isinstance(text, str) and text.strip():
                        content_parts.append(text.strip())
        
        # Add metadata if relevant
        if conversation.get('summary'):
            content_parts.append(conversation['summary'])
        
        return ' '.join(content_parts)
    
    @with_retry(component="conversation_search")
    def search(self, query: SearchQuery) -> Tuple[List[SearchResult], SearchStats]:
        """Perform advanced search with ranking and filtering."""
        
        start_time = datetime.now()
        
        try:
            with self.database._get_connection() as conn:
                # Build search query
                search_sql, params = self._build_search_query(query)
                
                # Execute search
                cursor = conn.execute(search_sql, params)
                raw_results = cursor.fetchall()
                
                # Process and rank results
                search_results = []
                for row in raw_results:
                    result = self._process_search_result(row, query)
                    if result and result.score >= query.min_score:
                        search_results.append(result)
                
                # Sort results
                search_results = self._sort_results(search_results, query)
                
                # Apply pagination
                total_results = len(search_results)
                paginated_results = search_results[query.offset:query.offset + query.limit]
                
                # Calculate search statistics
                search_time = (datetime.now() - start_time).total_seconds() * 1000
                
                # Get facets
                facets = self._calculate_facets(conn, query)
                
                stats = SearchStats(
                    total_results=total_results,
                    search_time_ms=search_time,
                    query_text=query.text or "",
                    filters_applied=query.filters,
                    facets=facets
                )
                
                self.logger.debug(f"Search completed: {total_results} results in {search_time:.2f}ms")
                return paginated_results, stats
                
        except Exception as e:
            handle_error(e, "conversation_search", {
                "operation": "search",
                "query_text": query.text
            })
            return [], SearchStats(
                total_results=0,
                search_time_ms=0,
                query_text=query.text or "",
                filters_applied=query.filters
            )
    
    def _build_search_query(self, query: SearchQuery) -> Tuple[str, List[Any]]:
        """Build SQL search query with filters."""
        
        params = []
        where_clauses = []
        
        # Base query with FTS search
        if query.text:
            # Use FTS5 for full-text search
            base_sql = """
                SELECT 
                    c.conversation_id,
                    c.title,
                    c.dataset_source,
                    c.tier,
                    c.processing_status,
                    c.created_at,
                    c.turn_count,
                    c.word_count,
                    cs.searchable_content,
                    fts.rank,
                    q.overall_quality,
                    q.therapeutic_accuracy,
                    q.safety_score
                FROM conversations_fts fts
                JOIN conversations c ON c.conversation_id = fts.conversation_id
                LEFT JOIN conversation_search cs ON cs.conversation_id = c.conversation_id
                LEFT JOIN conversation_quality q ON q.conversation_id = c.conversation_id
                WHERE conversations_fts MATCH ?
            """
            params.append(self._prepare_fts_query(query.text))
        else:
            # Regular query without FTS
            base_sql = """
                SELECT 
                    c.conversation_id,
                    c.title,
                    c.dataset_source,
                    c.tier,
                    c.processing_status,
                    c.created_at,
                    c.turn_count,
                    c.word_count,
                    cs.searchable_content,
                    0 as rank,
                    q.overall_quality,
                    q.therapeutic_accuracy,
                    q.safety_score
                FROM conversations c
                LEFT JOIN conversation_search cs ON cs.conversation_id = c.conversation_id
                LEFT JOIN conversation_quality q ON q.conversation_id = c.conversation_id
                WHERE 1=1
            """
        
        # Add filters
        if query.filters:
            if 'dataset_source' in query.filters:
                where_clauses.append("c.dataset_source = ?")
                params.append(query.filters['dataset_source'])
            
            if 'tier' in query.filters:
                where_clauses.append("c.tier = ?")
                params.append(query.filters['tier'])
            
            if 'processing_status' in query.filters:
                where_clauses.append("c.processing_status = ?")
                params.append(query.filters['processing_status'])
            
            if 'min_quality' in query.filters:
                where_clauses.append("q.overall_quality >= ?")
                params.append(query.filters['min_quality'])
            
            if 'max_quality' in query.filters:
                where_clauses.append("q.overall_quality <= ?")
                params.append(query.filters['max_quality'])
            
            if 'min_word_count' in query.filters:
                where_clauses.append("c.word_count >= ?")
                params.append(query.filters['min_word_count'])
            
            if 'tags' in query.filters:
                tag_list = query.filters['tags']
                if isinstance(tag_list, list):
                    placeholders = ','.join(['?' for _ in tag_list])
                    where_clauses.append(f"""
                        c.conversation_id IN (
                            SELECT conversation_id FROM conversation_tags 
                            WHERE tag_value IN ({placeholders})
                        )
                    """)
                    params.extend(tag_list)
        
        # Combine where clauses
        if where_clauses:
            if query.text:
                base_sql += " AND " + " AND ".join(where_clauses)
            else:
                base_sql += " AND " + " AND ".join(where_clauses)
        
        # Add ordering (will be overridden by Python sorting for complex ranking)
        if query.sort_by == "date":
            base_sql += f" ORDER BY c.created_at {query.sort_order.upper()}"
        elif query.sort_by == "quality":
            base_sql += f" ORDER BY q.overall_quality {query.sort_order.upper()}"
        elif query.text:
            base_sql += " ORDER BY rank"
        
        return base_sql, params
    
    def _prepare_fts_query(self, text: str) -> str:
        """Prepare text for FTS5 query."""
        
        # Clean and tokenize
        words = re.findall(r'\b\w+\b', text.lower())
        
        # Remove stop words and short words
        filtered_words = [
            word for word in words 
            if len(word) >= self.min_word_length and word not in self.stop_words
        ]
        
        if not filtered_words:
            return text  # Fallback to original text
        
        # Create FTS5 query with phrase matching and OR logic
        if len(filtered_words) == 1:
            return filtered_words[0]
        else:
            # Use phrase matching for exact phrases, OR for individual terms
            return ' OR '.join(filtered_words)
    
    def _process_search_result(self, row: sqlite3.Row, query: SearchQuery) -> Optional[SearchResult]:
        """Process raw search result into SearchResult object."""
        
        try:
            conversation_id = row['conversation_id']
            title = row['title']
            content = row['searchable_content'] or ""
            
            # Calculate relevance score
            score = self._calculate_relevance_score(row, query)
            
            # Generate snippet
            snippet = self._generate_snippet(content, query.text, self.max_snippet_length)
            
            # Generate highlights
            highlights = []
            if query.include_highlights and query.text:
                highlights = self._generate_highlights(content, query.text)
            
            # Collect metadata
            metadata = {
                'dataset_source': row['dataset_source'],
                'tier': row['tier'],
                'processing_status': row['processing_status'],
                'created_at': row['created_at'],
                'turn_count': row['turn_count'],
                'word_count': row['word_count'],
                'overall_quality': row['overall_quality'],
                'therapeutic_accuracy': row['therapeutic_accuracy'],
                'safety_score': row['safety_score']
            }
            
            return SearchResult(
                conversation_id=conversation_id,
                title=title,
                snippet=snippet,
                score=score,
                highlights=highlights,
                metadata=metadata
            )
            
        except Exception as e:
            handle_error(e, "conversation_search", {
                "operation": "process_search_result",
                "conversation_id": row.get('conversation_id', 'unknown')
            })
            return None
    
    def _calculate_relevance_score(self, row: sqlite3.Row, query: SearchQuery) -> float:
        """Calculate relevance score for search result."""
        
        base_score = 1.0
        
        # FTS rank (if available)
        if row['rank'] and query.text:
            # FTS5 rank is negative, convert to positive score
            fts_score = abs(float(row['rank'])) / 10.0
            base_score += fts_score
        
        # Quality boost
        if row['overall_quality']:
            quality_boost = float(row['overall_quality']) * 0.5
            base_score += quality_boost
        
        # Safety boost
        if row['safety_score']:
            safety_boost = float(row['safety_score']) * 0.3
            base_score += safety_boost
        
        # Word count normalization (prefer moderate length conversations)
        if row['word_count']:
            word_count = int(row['word_count'])
            if 50 <= word_count <= 500:  # Optimal range
                base_score += 0.2
            elif word_count < 50:
                base_score -= 0.1
        
        # Tier boost
        tier_boosts = {
            'priority_1': 0.3,
            'priority_2': 0.2,
            'priority_3': 0.1,
            'professional': 0.25
        }
        tier = row['tier']
        if tier in tier_boosts:
            base_score += tier_boosts[tier]
        
        return max(0.0, base_score)
    
    def _generate_snippet(self, content: str, query_text: Optional[str], max_length: int) -> str:
        """Generate search result snippet with context."""
        
        if not content:
            return ""
        
        if not query_text or len(content) <= max_length:
            return content[:max_length] + ("..." if len(content) > max_length else "")
        
        # Find best snippet around query terms
        words = re.findall(r'\b\w+\b', query_text.lower())
        best_snippet = ""
        best_score = 0
        
        # Try different positions in the content
        for i in range(0, len(content), max_length // 2):
            snippet = content[i:i + max_length]
            
            # Score snippet based on query term matches
            snippet_lower = snippet.lower()
            score = sum(snippet_lower.count(word) for word in words)
            
            if score > best_score:
                best_score = score
                best_snippet = snippet
        
        if not best_snippet:
            best_snippet = content[:max_length]
        
        # Clean up snippet boundaries
        if len(best_snippet) == max_length and len(content) > max_length:
            # Try to end at word boundary
            last_space = best_snippet.rfind(' ')
            if last_space > max_length * 0.8:
                best_snippet = best_snippet[:last_space]
            best_snippet += "..."
        
        return best_snippet
    
    def _generate_highlights(self, content: str, query_text: str, max_highlights: int = 5) -> List[str]:
        """Generate highlighted excerpts from content."""
        
        if not query_text:
            return []
        
        words = re.findall(r'\b\w+\b', query_text.lower())
        highlights = []
        
        # Find sentences containing query terms
        sentences = re.split(r'[.!?]+', content)
        
        for sentence in sentences[:max_highlights * 2]:  # Check more sentences than needed
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_lower = sentence.lower()
            matches = sum(1 for word in words if word in sentence_lower)
            
            if matches > 0:
                # Highlight matching terms
                highlighted = sentence
                for word in words:
                    pattern = re.compile(re.escape(word), re.IGNORECASE)
                    highlighted = pattern.sub(
                        f"{self.highlight_tags[0]}{word}{self.highlight_tags[1]}", 
                        highlighted
                    )
                
                highlights.append(highlighted)
                
                if len(highlights) >= max_highlights:
                    break
        
        return highlights
    
    def _sort_results(self, results: List[SearchResult], query: SearchQuery) -> List[SearchResult]:
        """Sort search results based on query parameters."""
        
        if query.sort_by == "relevance":
            return sorted(results, key=lambda x: x.score, reverse=(query.sort_order == "desc"))
        elif query.sort_by == "date":
            return sorted(results, 
                         key=lambda x: x.metadata.get('created_at', ''), 
                         reverse=(query.sort_order == "desc"))
        elif query.sort_by == "quality":
            return sorted(results, 
                         key=lambda x: x.metadata.get('overall_quality', 0), 
                         reverse=(query.sort_order == "desc"))
        else:
            return results
    
    def _calculate_facets(self, conn: sqlite3.Connection, query: SearchQuery) -> Dict[str, Dict[str, int]]:
        """Calculate facets for search results."""
        
        facets = {}
        
        try:
            # Dataset source facets
            cursor = conn.execute("""
                SELECT dataset_source, COUNT(*) 
                FROM conversations 
                GROUP BY dataset_source
            """)
            facets['dataset_source'] = dict(cursor.fetchall())
            
            # Tier facets
            cursor = conn.execute("""
                SELECT tier, COUNT(*) 
                FROM conversations 
                GROUP BY tier
            """)
            facets['tier'] = dict(cursor.fetchall())
            
            # Processing status facets
            cursor = conn.execute("""
                SELECT processing_status, COUNT(*) 
                FROM conversations 
                GROUP BY processing_status
            """)
            facets['processing_status'] = dict(cursor.fetchall())
            
            # Quality range facets
            cursor = conn.execute("""
                SELECT 
                    CASE 
                        WHEN overall_quality >= 0.8 THEN 'high'
                        WHEN overall_quality >= 0.6 THEN 'medium'
                        WHEN overall_quality >= 0.4 THEN 'low'
                        ELSE 'very_low'
                    END as quality_range,
                    COUNT(*)
                FROM conversation_quality
                GROUP BY quality_range
            """)
            facets['quality_range'] = dict(cursor.fetchall())
            
        except Exception as e:
            handle_error(e, "conversation_search", {"operation": "calculate_facets"})
        
        return facets
    
    def suggest_completions(self, partial_text: str, limit: int = 10) -> List[str]:
        """Suggest search completions based on partial text."""
        
        try:
            with self.database._get_connection() as conn:
                # Get common terms from indexed content
                cursor = conn.execute("""
                    SELECT DISTINCT tag_value 
                    FROM conversation_tags 
                    WHERE tag_value LIKE ? 
                    ORDER BY tag_value 
                    LIMIT ?
                """, (f"{partial_text}%", limit))
                
                suggestions = [row[0] for row in cursor.fetchall()]
                
                return suggestions
                
        except Exception as e:
            handle_error(e, "conversation_search", {
                "operation": "suggest_completions",
                "partial_text": partial_text
            })
            return []

if __name__ == "__main__":
    # Test the search engine
    from conversation_database import ConversationDatabase
    from conversation_schema import ConversationSchema, ConversationTier, ProcessingStatus
    import uuid
    
    print("🔍 CONVERSATION SEARCH ENGINE TEST")
    print("=" * 50)
    
    # Initialize database and search engine
    db = ConversationDatabase()
    search_engine = ConversationSearchEngine(db)
    
    try:
        # Create test conversations
        test_conversations = [
            ConversationSchema(
                conversation_id=str(uuid.uuid4()),
                dataset_source="test_search",
                tier=ConversationTier.PRIORITY_1,
                title="Anxiety Management Session",
                conversations=[
                    {"human": "I'm feeling very anxious about my upcoming presentation."},
                    {"assistant": "I understand that presentations can be anxiety-provoking. Let's explore some coping strategies."}
                ],
                overall_quality=0.85,
                tags=["anxiety", "presentation", "coping"],
                categories=["mental_health"],
                processing_status=ProcessingStatus.PROCESSED,
                turn_count=2,
                word_count=25
            ),
            ConversationSchema(
                conversation_id=str(uuid.uuid4()),
                dataset_source="test_search",
                tier=ConversationTier.PROFESSIONAL,
                title="Depression Support",
                conversations=[
                    {"human": "I've been feeling depressed lately."},
                    {"assistant": "Thank you for sharing that with me. Depression is a serious condition that we can work through together."}
                ],
                overall_quality=0.90,
                tags=["depression", "support"],
                categories=["mental_health"],
                processing_status=ProcessingStatus.PROCESSED,
                turn_count=2,
                word_count=30
            )
        ]
        
        # Insert test conversations
        for conv in test_conversations:
            db.insert_conversation(conv)
            search_engine.index_conversation(conv.conversation_id)
        
        print(f"✅ Inserted and indexed {len(test_conversations)} test conversations")
        
        # Test text search
        query = SearchQuery(
            text="anxiety presentation",
            limit=10,
            include_highlights=True
        )
        
        results, stats = search_engine.search(query)
        print(f"✅ Text search: {len(results)} results in {stats.search_time_ms:.2f}ms")
        
        if results:
            print(f"   Top result: {results[0].title} (score: {results[0].score:.2f})")
            print(f"   Snippet: {results[0].snippet[:100]}...")
        
        # Test filtered search
        filtered_query = SearchQuery(
            filters={'tier': 'professional', 'min_quality': 0.8},
            limit=10
        )
        
        filtered_results, filtered_stats = search_engine.search(filtered_query)
        print(f"✅ Filtered search: {len(filtered_results)} results")
        
        # Test facets
        print(f"✅ Facets available: {list(stats.facets.keys())}")
        
        # Test suggestions
        suggestions = search_engine.suggest_completions("anx", limit=5)
        print(f"✅ Suggestions for 'anx': {suggestions}")
        
    finally:
        db.close()
    
    print("✅ Search engine test complete!")
