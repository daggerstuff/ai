"""
Metadata Parser

Parses and normalizes metadata from different source formats into
unified DatasetSource objects.
"""

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

from ai.journal_dataset_research.models.dataset_models import DatasetSource

logger = logging.getLogger(__name__)


class MetadataParser:
    """Parser for normalizing metadata from different source formats."""

    def parse(
        self,
        raw_data: Dict[str, Any],
        source_type: str,
        discovery_method: str,
    ) -> Optional[DatasetSource]:
        """
        Parse raw metadata into a DatasetSource.

        Args:
            raw_data: Raw metadata dictionary
            source_type: Type of source (journal, repository, clinical_trial, training_material)
            discovery_method: Method used to discover this source

        Returns:
            DatasetSource object or None if parsing fails
        """
        try:
            # Extract common fields
            title = self._extract_title(raw_data)
            authors = self._extract_authors(raw_data)
            pub_date = self._extract_date(raw_data)
            doi = self._extract_doi(raw_data)
            url = self._extract_url(raw_data)
            abstract = self._extract_abstract(raw_data)
            keywords = self._extract_keywords(raw_data)

            # Determine data availability
            data_availability = self._detect_data_availability(raw_data, abstract, title)

            # Determine open access status
            open_access = self._detect_open_access(raw_data, source_type)

            # Generate source ID
            source_id = self._generate_source_id(raw_data, source_type, doi, title)

            return DatasetSource(
                source_id=source_id,
                title=title,
                authors=authors,
                publication_date=pub_date,
                source_type=source_type,
                url=url,
                doi=doi,
                abstract=abstract,
                keywords=keywords,
                open_access=open_access,
                data_availability=data_availability,
                discovery_date=datetime.now(),
                discovery_method=discovery_method,
            )

        except Exception as e:
            logger.warning(f"Error parsing metadata: {e}")
            return None

    def _extract_title(self, data: Dict[str, Any]) -> str:
        """Extract title from metadata."""
        # Try common title fields
        for field in ["title", "Title", "name", "Name", "briefTitle"]:
            if field in data:
                value = data[field]
                if isinstance(value, str):
                    return value
                elif isinstance(value, list) and value:
                    return str(value[0])

        # Try nested structures
        if "metadata" in data:
            return self._extract_title(data["metadata"])
        if "bibjson" in data:
            return self._extract_title(data["bibjson"])

        return "Untitled"

    def _extract_authors(self, data: Dict[str, Any]) -> List[str]:
        """Extract authors from metadata."""
        authors = []

        # Try common author fields
        for field in ["authors", "Authors", "author", "Author", "creators", "Creators"]:
            if field in data:
                author_list = data[field]
                if isinstance(author_list, list):
                    for author in author_list:
                        if isinstance(author, dict):
                            name = (
                                author.get("name")
                                or author.get("Name")
                                or author.get("fullName")
                                or f'{author.get("firstName", "")} {author.get("lastName", "")}'.strip()
                            )
                            if name:
                                authors.append(name)
                        elif isinstance(author, str):
                            authors.append(author)
                elif isinstance(author_list, str):
                    authors.append(author_list)

        # Try nested structures
        if "metadata" in data:
            authors.extend(self._extract_authors(data["metadata"]))
        if "bibjson" in data:
            authors.extend(self._extract_authors(data["bibjson"]))

        return list(dict.fromkeys(authors))  # Remove duplicates while preserving order

    def _extract_date(self, data: Dict[str, Any]) -> datetime:
        """Extract publication date from metadata."""
        # Try common date fields
        for field in [
            "publication_date",
            "publicationDate",
            "datePublished",
            "year",
            "Year",
            "date",
            "Date",
            "pubDate",
        ]:
            if field in data:
                date_value = data[field]
                try:
                    if isinstance(date_value, str):
                        # Try ISO format
                        return datetime.fromisoformat(date_value.replace("Z", "+00:00"))
                    elif isinstance(date_value, (int, float)):
                        # Assume it's a year
                        return datetime(int(date_value), 1, 1)
                    elif isinstance(date_value, dict):
                        # Try to extract year, month, day
                        year = date_value.get("year") or date_value.get("Year")
                        month = date_value.get("month") or date_value.get("Month", 1)
                        day = date_value.get("day") or date_value.get("Day", 1)
                        if year:
                            return datetime(int(year), int(month), int(day))
                except (ValueError, TypeError, AttributeError):
                    continue

        # Try nested structures
        if "metadata" in data:
            date = self._extract_date(data["metadata"])
            if date != datetime.now():
                return date

        return datetime.now()

    def _extract_doi(self, data: Dict[str, Any]) -> Optional[str]:
        """Extract DOI from metadata."""
        # Try common DOI fields
        if "doi" in data:
            doi = data["doi"]
            if isinstance(doi, str):
                return doi
            elif isinstance(doi, dict):
                return doi.get("id") or doi.get("value")

        # Try identifier lists
        if "identifier" in data:
            identifier = data["identifier"]
            if isinstance(identifier, list):
                for ident in identifier:
                    if isinstance(ident, dict):
                        if ident.get("type") == "doi":
                            return ident.get("id") or ident.get("value")
            elif isinstance(identifier, dict):
                if identifier.get("type") == "doi":
                    return identifier.get("id") or identifier.get("value")

        # Try nested structures
        if "metadata" in data:
            return self._extract_doi(data["metadata"])
        if "bibjson" in data:
            return self._extract_doi(data["bibjson"])

        return None

    def _extract_url(self, data: Dict[str, Any]) -> str:
        """Extract URL from metadata."""
        # Try common URL fields
        for field in ["url", "URL", "link", "Link", "html", "self"]:
            if field in data:
                url = data[field]
                if isinstance(url, str):
                    return url
                elif isinstance(url, dict):
                    return url.get("url") or url.get("href") or url.get("value")

        # Try link lists
        if "links" in data:
            links = data["links"]
            if isinstance(links, dict):
                for link_field in ["html", "self", "url", "doi"]:
                    if link_field in links:
                        return links[link_field]

        # Try nested structures
        if "metadata" in data:
            url = self._extract_url(data["metadata"])
            if url != "https://example.com":
                return url

        # Fallback: construct from DOI
        doi = self._extract_doi(data)
        if doi:
            return f"https://doi.org/{doi}"

        return "https://example.com"

    def _extract_abstract(self, data: Dict[str, Any]) -> str:
        """Extract abstract/description from metadata."""
        # Try common abstract fields
        for field in [
            "abstract",
            "Abstract",
            "description",
            "Description",
            "summary",
            "Summary",
            "briefSummary",
        ]:
            if field in data:
                abstract = data[field]
                if isinstance(abstract, str):
                    return abstract
                elif isinstance(abstract, list) and abstract:
                    return str(abstract[0])

        # Try nested structures
        if "metadata" in data:
            return self._extract_abstract(data["metadata"])
        if "bibjson" in data:
            return self._extract_abstract(data["bibjson"])

        return ""

    def _extract_keywords(self, data: Dict[str, Any]) -> List[str]:
        """Extract keywords from metadata."""
        keywords = []

        # Try common keyword fields
        for field in ["keywords", "Keywords", "tags", "Tags", "subjects", "Subjects"]:
            if field in data:
                keyword_list = data[field]
                if isinstance(keyword_list, list):
                    for keyword in keyword_list:
                        if isinstance(keyword, str):
                            keywords.append(keyword)
                        elif isinstance(keyword, dict):
                            kw = keyword.get("name") or keyword.get("value") or keyword.get("term")
                            if kw:
                                keywords.append(kw)
                elif isinstance(keyword_list, str):
                    keywords.append(keyword_list)

        # Try nested structures
        if "metadata" in data:
            keywords.extend(self._extract_keywords(data["metadata"]))

        return list(dict.fromkeys(keywords))  # Remove duplicates

    def _detect_data_availability(self, data: Dict[str, Any], abstract: str, title: str) -> str:
        """Detect data availability from metadata and text."""
        text = (abstract + " " + title).lower()

        # Check metadata fields
        if "dataAvailability" in data or "data_availability" in data:
            availability = data.get("dataAvailability") or data.get("data_availability", "")
            if isinstance(availability, str):
                availability_lower = availability.lower()
                if "available" in availability_lower:
                    return "available"
                elif "upon request" in availability_lower or "upon_request" in availability_lower:
                    return "upon_request"
                elif "restricted" in availability_lower:
                    return "restricted"

        # Check text for indicators
        if any(
            phrase in text
            for phrase in [
                "data available",
                "dataset available",
                "supplementary data",
                "data repository",
                "data deposited",
            ]
        ):
            return "available"

        if any(
            phrase in text
            for phrase in [
                "data upon request",
                "data available upon request",
                "contact author",
                "data available from",
            ]
        ):
            return "upon_request"

        if any(
            phrase in text
            for phrase in [
                "restricted access",
                "data not available",
                "confidential",
                "proprietary",
            ]
        ):
            return "restricted"

        return "unknown"

    def _detect_open_access(self, data: Dict[str, Any], source_type: str) -> bool:
        """Detect if source is open access."""
        # Check explicit field
        if "openAccess" in data or "open_access" in data:
            return bool(data.get("openAccess") or data.get("open_access"))

        # Repository sources are typically open access
        if source_type in ["repository", "clinical_trial"]:
            return True

        # Check license
        license_info = data.get("license") or data.get("License")
        if license_info:
            if isinstance(license_info, str):
                license_lower = license_info.lower()
                if "open" in license_lower or "cc" in license_lower:
                    return True

        return False

    def _generate_source_id(
        self, data: Dict[str, Any], source_type: str, doi: Optional[str], title: str
    ) -> str:
        """Generate a unique source ID."""
        # Try to use existing ID
        for field in ["id", "ID", "identifier", "Identifier", "nctId"]:
            if field in data:
                id_value = data[field]
                if isinstance(id_value, str):
                    return f"{source_type}_{id_value}"

        # Use DOI if available
        if doi:
            # Clean DOI
            doi_clean = doi.replace("https://doi.org/", "").replace("doi:", "")
            return f"{source_type}_{doi_clean.replace('/', '_')}"

        # Fallback to hash of title
        return f"{source_type}_{abs(hash(title))}"

