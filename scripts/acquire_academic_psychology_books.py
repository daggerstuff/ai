#!/usr/bin/env python3
"""
Acquire Psychology and Therapy Books from Academic Publishers
Automated pipeline for sourcing psychology/therapy books from academic publishers and repositories
"""
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from urllib.parse import urljoin, urlparse
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import time
from dataclasses import dataclass
from enum import Enum

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class BookMetadata:
    """Structured metadata for acquired psychology books"""
    title: str
    authors: List[str]
    publisher: str
    publication_year: int
    isbn: Optional[str]
    url: str
    source: str
    content: Optional[str] = None
    chapter_count: Optional[int] = None
    page_count: Optional[int] = None
    subject_areas: List[str] = None
    confidence_score: float = 0.0

class SourceType(Enum):
    GOOGLE_SCHOLAR = "google_scholar"
    PUBMED = "pubmed"
    JSTOR = "jstor"
    SPRINGER = "springer"
    OXFORD_UNIV_PRESS = "oxford_univ_press"
    CAMBRIDGE_UNIV_PRESS = "cambridge_univ_press"
    APA_PUBLISHING = "apa_publishing"
    LOCAL_LIBRARY = "local_library"

class AcademicBookAcquisition:
    """Acquire psychology and therapy books from academic sources"""
    
    def __init__(self, output_dir: Optional[Path] = None):
        self.output_dir = output_dir or Path("ai/data/acquired_books")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure sources with their specific patterns
        self.sources = {
            SourceType.GOOGLE_SCHOLAR: {
                "base_url": "https://scholar.google.com",
                "search_endpoint": "/scholar",
                "headers": {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                },
                "query_params": {
                    "hl": "en",
                    "as_sdt": "0,5"
                }
            },
            SourceType.PUBMED: {
                "base_url": "https://pubmed.ncbi.nlm.nih.gov",
                "search_endpoint": "/",
                "headers": {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }
            },
            SourceType.JSTOR: {
                "base_url": "https://www.jstor.org",
                "search_endpoint": "/search",
                "headers": {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }
            },
            SourceType.SPRINGER: {
                "base_url": "https://link.springer.com",
                "search_endpoint": "/search",
                "headers": {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }
            },
            SourceType.OXFORD_UNIV_PRESS: {
                "base_url": "https://academic.oup.com",
                "search_endpoint": "/search-results",
                "headers": {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }
            },
            SourceType.CAMBRIDGE_UNIV_PRESS: {
                "base_url": "https://www.cambridge.org",
                "search_endpoint": "/core/services/aop-cambridge-core/contact",
                "headers": {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }
            },
            SourceType.APA_PUBLISHING: {
                "base_url": "https://www.apa.org",
                "search_endpoint": "/pubs/books",
                "headers": {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
                }
            }
        }
        
        # Define psychology-specific search terms
        self.search_terms = [
            "psychology textbook",
            "clinical psychology",
            "cognitive behavioral therapy",
            "DBT therapy",
            "psychodynamic therapy",
            "trauma-informed care",
            "mental health counseling",
            "psychotherapy techniques",
            "DSM-5 diagnostic criteria",
            "psychological assessment",
            "therapeutic alliance",
            "empathy in therapy",
            "cultural competence in therapy",
            "mental health interventions",
            "evidence-based therapy",
            "psychology of trauma",
            "addiction counseling",
            "family systems therapy",
            "mindfulness-based therapy",
            "psychology of depression",
            "anxiety disorders treatment",
            "personality disorders",
            "psychology of suicide",
            "crisis intervention",
            "ethical issues in therapy"
        ]
        
        # Define publishers known for academic psychology books
        self.academic_publishers = [
            "Oxford University Press",
            "Cambridge University Press",
            "American Psychological Association",
            "Springer Nature",
            "Wiley",
            "Routledge",
            "Guilford Press",
            "SAGE Publications",
            "Pearson",
            "McGraw-Hill",
            "Elsevier",
            "Taylor & Francis"
        ]

    def search_google_scholar(self, query: str, max_results: int = 10) -> List[BookMetadata]:
        """Search Google Scholar for psychology books"""
        logger.info(f"🔍 Searching Google Scholar for: {query}")
        
        results = []
        url = urljoin(self.sources[SourceType.GOOGLE_SCHOLAR]["base_url"], 
                     self.sources[SourceType.GOOGLE_SCHOLAR]["search_endpoint"])
        
        params = self.sources[SourceType.GOOGLE_SCHOLAR]["query_params"].copy()
        params["q"] = query
        params["num"] = max_results
        
        try:
            response = requests.get(url, params=params, 
                                  headers=self.sources[SourceType.GOOGLE_SCHOLAR]["headers"],
                                  timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract book results
            for item in soup.select('.gs_ri')[:max_results]:
                try:
                    title_elem = item.select_one('.gs_rt a')
                    if not title_elem:
                        continue
                        
                    title = title_elem.get_text().strip()
                    url = title_elem['href'] if title_elem.get('href') else ''
                    
                    # Extract author and publication info
                    author_pub = item.select_one('.gs_a')
                    author_pub_text = author_pub.get_text() if author_pub else ""
                    
                    # Extract authors (everything before the first dash or comma)
                    authors = []
                    if " - " in author_pub_text:
                        authors_part = author_pub_text.split(" - ")[0]
                    elif "," in author_pub_text:
                        authors_part = author_pub_text.split(",")[0]
                    else:
                        authors_part = author_pub_text
                    
                    # Split by comma or "and" to get individual authors
                    if " and " in authors_part:
                        authors = [a.strip() for a in authors_part.split(" and ")]
                    else:
                        authors = [a.strip() for a in authors_part.split(",") if a.strip()]
                    
                    # Extract publication year (last 4 digits in the string)
                    year = None
                    for part in author_pub_text.split():
                        if len(part) == 4 and part.isdigit():
                            year = int(part)
                            break
                    
                    # Extract publisher (look for known academic publishers)
                    publisher = None
                    for pub in self.academic_publishers:
                        if pub.lower() in author_pub_text.lower():
                            publisher = pub
                            break
                    
                    # Create metadata
                    metadata = BookMetadata(
                        title=title,
                        authors=authors,
                        publisher=publisher or "Unknown",
                        publication_year=year or 0,
                        isbn=None,
                        url=url,
                        source=SourceType.GOOGLE_SCHOLAR.value,
                        subject_areas=[query],
                        confidence_score=0.8
                    )
                    
                    results.append(metadata)
                    
                except Exception as e:
                    logger.warning(f"Error parsing Google Scholar result: {e}")
                    continue
                    
            logger.info(f"✅ Found {len(results)} results from Google Scholar for '{query}'")
            
        except Exception as e:
            logger.error(f"Error searching Google Scholar: {e}")
            
        return results

    def search_springer(self, query: str, max_results: int = 10) -> List[BookMetadata]:
        """Search Springer for psychology books"""
        logger.info(f"🔍 Searching Springer for: {query}")
        
        results = []
        url = urljoin(self.sources[SourceType.SPRINGER]["base_url"], 
                     self.sources[SourceType.SPRINGER]["search_endpoint"])
        
        params = {
            "query": query,
            "facet-content-type": "Book",
            "sortBy": "relevance"
        }
        
        try:
            response = requests.get(url, params=params, 
                                  headers=self.sources[SourceType.SPRINGER]["headers"],
                                  timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract book results
            for item in soup.select('.result-item')[:max_results]:
                try:
                    title_elem = item.select_one('.title')
                    if not title_elem:
                        continue
                        
                    title = title_elem.get_text().strip()
                    
                    # Extract URL
                    link_elem = item.select_one('.title a')
                    url = link_elem['href'] if link_elem and link_elem.get('href') else ''
                    
                    # Extract authors
                    authors_elem = item.select_one('.authors')
                    authors = []
                    if authors_elem:
                        authors_text = authors_elem.get_text().strip()
                        authors = [a.strip() for a in authors_text.split(',') if a.strip()]
                    
                    # Extract publisher and year
                    pub_year_elem = item.select_one('.publication-info')
                    pub_year_text = pub_year_elem.get_text().strip() if pub_year_elem else ""
                    
                    publisher = None
                    year = None
                    
                    # Extract year (last 4 digits)
                    for part in pub_year_text.split():
                        if len(part) == 4 and part.isdigit():
                            year = int(part)
                            break
                    
                    # Extract publisher
                    for pub in self.academic_publishers:
                        if pub.lower() in pub_year_text.lower():
                            publisher = pub
                            break
                    
                    # Create metadata
                    metadata = BookMetadata(
                        title=title,
                        authors=authors,
                        publisher=publisher or "Springer Nature",
                        publication_year=year or 0,
                        isbn=None,
                        url=url,
                        source=SourceType.SPRINGER.value,
                        subject_areas=[query],
                        confidence_score=0.9
                    )
                    
                    results.append(metadata)
                    
                except Exception as e:
                    logger.warning(f"Error parsing Springer result: {e}")
                    continue
                    
            logger.info(f"✅ Found {len(results)} results from Springer for '{query}'")
            
        except Exception as e:
            logger.error(f"Error searching Springer: {e}")
            
        return results

    def search_oxford(self, query: str, max_results: int = 10) -> List[BookMetadata]:
        """Search Oxford University Press for psychology books"""
        logger.info(f"🔍 Searching Oxford University Press for: {query}")
        
        results = []
        url = urljoin(self.sources[SourceType.OXFORD_UNIV_PRESS]["base_url"], 
                     self.sources[SourceType.OXFORD_UNIV_PRESS]["search_endpoint"])
        
        params = {
            "q": query,
            "facet": "content-type:book",
            "sort": "relevance"
        }
        
        try:
            response = requests.get(url, params=params, 
                                  headers=self.sources[SourceType.OXFORD_UNIV_PRESS]["headers"],
                                  timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract book results
            for item in soup.select('.result-item')[:max_results]:
                try:
                    title_elem = item.select_one('.title a')
                    if not title_elem:
                        continue
                        
                    title = title_elem.get_text().strip()
                    url = title_elem['href'] if title_elem.get('href') else ''
                    
                    # Extract authors
                    authors_elem = item.select_one('.author')
                    authors = []
                    if authors_elem:
                        authors_text = authors_elem.get_text().strip()
                        authors = [a.strip() for a in authors_text.split(',') if a.strip()]
                    
                    # Extract publisher and year
                    pub_info_elem = item.select_one('.publication-info')
                    pub_info_text = pub_info_elem.get_text().strip() if pub_info_elem else ""
                    
                    publisher = "Oxford University Press"
                    year = None
                    
                    # Extract year
                    for part in pub_info_text.split():
                        if len(part) == 4 and part.isdigit():
                            year = int(part)
                            break
                    
                    # Create metadata
                    metadata = BookMetadata(
                        title=title,
                        authors=authors,
                        publisher=publisher,
                        publication_year=year or 0,
                        isbn=None,
                        url=url,
                        source=SourceType.OXFORD_UNIV_PRESS.value,
                        subject_areas=[query],
                        confidence_score=0.9
                    )
                    
                    results.append(metadata)
                    
                except Exception as e:
                    logger.warning(f"Error parsing Oxford result: {e}")
                    continue
                    
            logger.info(f"✅ Found {len(results)} results from Oxford for '{query}'")
            
        except Exception as e:
            logger.error(f"Error searching Oxford: {e}")
            
        return results

    def search_apa(self, query: str, max_results: int = 10) -> List[BookMetadata]:
        """Search APA Publishing for psychology books"""
        logger.info(f"🔍 Searching APA Publishing for: {query}")
        
        results = []
        url = urljoin(self.sources[SourceType.APA_PUBLISHING]["base_url"], 
                     self.sources[SourceType.APA_PUBLISHING]["search_endpoint"])
        
        try:
            response = requests.get(url, 
                                  headers=self.sources[SourceType.APA_PUBLISHING]["headers"],
                                  timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract book results from APA's book page
            for item in soup.select('.book-item')[:max_results]:
                try:
                    title_elem = item.select_one('.book-title a')
                    if not title_elem:
                        continue
                        
                    title = title_elem.get_text().strip()
                    url = title_elem['href'] if title_elem.get('href') else ''
                    
                    # Extract authors
                    authors_elem = item.select_one('.book-authors')
                    authors = []
                    if authors_elem:
                        authors_text = authors_elem.get_text().strip()
                        authors = [a.strip() for a in authors_text.split(',') if a.strip()]
                    
                    # Extract publisher and year
                    pub_info_elem = item.select_one('.book-pub-info')
                    pub_info_text = pub_info_elem.get_text().strip() if pub_info_elem else ""
                    
                    publisher = "American Psychological Association"
                    year = None
                    
                    # Extract year
                    for part in pub_info_text.split():
                        if len(part) == 4 and part.isdigit():
                            year = int(part)
                            break
                    
                    # Create metadata
                    metadata = BookMetadata(
                        title=title,
                        authors=authors,
                        publisher=publisher,
                        publication_year=year or 0,
                        isbn=None,
                        url=url,
                        source=SourceType.APA_PUBLISHING.value,
                        subject_areas=[query],
                        confidence_score=0.95
                    )
                    
                    results.append(metadata)
                    
                except Exception as e:
                    logger.warning(f"Error parsing APA result: {e}")
                    continue
                    
            logger.info(f"✅ Found {len(results)} results from APA for '{query}'")
            
        except Exception as e:
            logger.error(f"Error searching APA: {e}")
            
        return results

    def acquire_books_from_sources(self, max_books_per_term: int = 5) -> List[BookMetadata]:
        """Acquire books from all academic sources"""
        logger.info("🚀 Starting academic psychology book acquisition...")
        
        all_books = []
        
        # Search each source for each search term
        for term in self.search_terms:
            logger.info(f"🔍 Processing search term: {term}")
            
            # Search each source
            sources_to_search = [
                SourceType.GOOGLE_SCHOLAR,
                SourceType.SPRINGER,
                SourceType.OXFORD_UNIV_PRESS,
                SourceType.APA_PUBLISHING
            ]
            
            for source in sources_to_search:
                try:
                    if source == SourceType.GOOGLE_SCHOLAR:
                        books = self.search_google_scholar(term, max_books_per_term)
                    elif source == SourceType.SPRINGER:
                        books = self.search_springer(term, max_books_per_term)
                    elif source == SourceType.OXFORD_UNIV_PRESS:
                        books = self.search_oxford(term, max_books_per_term)
                    elif source == SourceType.APA_PUBLISHING:
                        books = self.search_apa(term, max_books_per_term)
                    else:
                        continue
                    
                    # Add books to results
                    for book in books:
                        # Avoid duplicates by title and author
                        is_duplicate = False
                        for existing_book in all_books:
                            if (existing_book.title.lower() == book.title.lower() and 
                                existing_book.authors and book.authors and
                                existing_book.authors[0].lower() == book.authors[0].lower()):
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            all_books.append(book)
                            
                    # Be respectful with rate limiting
                    time.sleep(2)
                    
                except Exception as e:
                    logger.error(f"Error searching {source.value} for '{term}': {e}")
                    continue
        
        logger.info(f"✅ Acquired {len(all_books)} unique psychology books from academic sources")
        return all_books

    def save_books_to_json(self, books: List[BookMetadata], filename: str = "academic_psychology_books.json"):
        """Save acquired books to JSON file"""
        output_file = self.output_dir / filename
        books_data = []
        
        for book in books:
            book_dict = {
                "title": book.title,
                "authors": book.authors,
                "publisher": book.publisher,
                "publication_year": book.publication_year,
                "isbn": book.isbn,
                "url": book.url,
                "source": book.source,
                "subject_areas": book.subject_areas,
                "confidence_score": book.confidence_score
            }
            books_data.append(book_dict)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(books_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Saved {len(books)} books to {output_file}")
        return output_file

    def create_summary(self, books: List[BookMetadata]) -> Dict[str, Any]:
        """Create summary of acquired books"""
        summary = {
            "total_books": len(books),
            "sources": {},
            "publishers": {},
            "years": {},
            "subject_areas": {},
            "average_confidence": sum(book.confidence_score for book in books) / len(books) if books else 0
        }
        
        for book in books:
            # Count by source
            source = book.source
            summary["sources"][source] = summary["sources"].get(source, 0) + 1
            
            # Count by publisher
            publisher = book.publisher
            summary["publishers"][publisher] = summary["publishers"].get(publisher, 0) + 1
            
            # Count by year
            year = book.publication_year
            if year > 0:
                summary["years"][year] = summary["years"].get(year, 0) + 1
            
            # Count by subject area
            for subject in book.subject_areas:
                summary["subject_areas"][subject] = summary["subject_areas"].get(subject, 0) + 1
        
        # Sort publishers by count
        summary["publishers"] = dict(sorted(summary["publishers"].items(), 
                                           key=lambda x: x[1], reverse=True))
        
        # Sort years by count
        summary["years"] = dict(sorted(summary["years"].items(), 
                                      key=lambda x: x[1], reverse=True))
        
        return summary

    def acquire_all_books(self) -> Dict[str, Any]:
        """Main method to acquire all books and create summary"""
        books = self.acquire_books_from_sources()
        
        # Save books to JSON
        json_file = self.save_books_to_json(books)
        
        # Create summary
        summary = self.create_summary(books)
        
        # Save summary
        summary_file = self.output_dir / "acquisition_summary.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"\n📊 Acquisition Summary:")
        logger.info(f"   Total books: {summary['total_books']}")
        logger.info(f"   Average confidence: {summary['average_confidence']:.2f}")
        logger.info(f"   Sources: {summary['sources']}")
        logger.info(f"   Top publishers: {list(summary['publishers'].items())[:5]}")
        logger.info(f"   Top subject areas: {list(summary['subject_areas'].items())[:5]}")
        logger.info(f"   Summary saved to: {summary_file}")
        
        return {
            "books": books,
            "summary": summary,
            "json_file": json_file,
            "summary_file": summary_file
        }


def main():
    """Main execution"""
    acquisitor = AcademicBookAcquisition()
    results = acquisitor.acquire_all_books()
    
    logger.info("\n✅ Academic psychology book acquisition complete!")
    logger.info(f"📁 Books saved to: {acquisitor.output_dir}")
    
    return results


if __name__ == "__main__":
    main()