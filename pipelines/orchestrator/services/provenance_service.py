#!/usr/bin/env python3
"""
Provenance Service

Service layer for managing dataset provenance metadata.
Provides CRUD operations, S3 integration, and audit logging.

Related Documentation:
- Schema: docs/governance/provenance_schema.json
- Storage Plan: docs/governance/provenance_storage_plan.md
- Schema Implementation: ai/pipelines/orchestrator/schemas/provenance_schema.py
"""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import asyncpg
import structlog
from tenacity import retry, stop_after_attempt, wait_exponential

from ..schemas.provenance_schema import (
    ProvenanceRecord,
)

logger = structlog.get_logger(__name__)


class ProvenanceService:
    """
    Service for managing dataset provenance metadata.

    Handles:
    - CRUD operations for provenance records
    - Database persistence (PostgreSQL/Supabase)
    - S3 file storage for audit trail
    - Audit logging
    - Query and reporting
    """

    def __init__(
        self,
        database_url: Optional[str] = None,
        s3_bucket: Optional[str] = None,
        s3_region: Optional[str] = None,
        s3_endpoint_url: Optional[str] = None,
    ):
        """
        Initialize ProvenanceService.

        Args:
            database_url: PostgreSQL connection string (defaults to env var)
            s3_bucket: S3 bucket name for file storage (defaults to env var)
            s3_region: S3 region (defaults to env var)
            s3_endpoint_url: S3 endpoint URL (defaults to env var)
        """
        self.database_url = (
            database_url or os.getenv("DATABASE_URL") or os.getenv("SUPABASE_DB_URL")
        )
        self.s3_bucket = s3_bucket or os.getenv("OVH_S3_BUCKET", os.getenv("S3_BUCKET", "pixel-data"))
        self.s3_region = s3_region or os.getenv("OVH_S3_REGION", os.getenv("S3_REGION", "us-east-1"))
        self.s3_endpoint_url = s3_endpoint_url or os.getenv("OVH_S3_ENDPOINT", os.getenv("S3_ENDPOINT_URL", "https://s3.us-east-va.io.cloud.ovh.us"))
        self.pg_pool: Optional[asyncpg.Pool] = None
        self.is_connected = False
        self._s3_client = None

        if not self.database_url:
            raise ValueError(
                "database_url required. Set DATABASE_URL or SUPABASE_DB_URL env var"
            )

    async def connect(self) -> bool:
        """
        Connect to PostgreSQL database.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            logger.info("Connecting to PostgreSQL for provenance service")

            self.pg_pool = await asyncpg.create_pool(
                self.database_url,
                min_size=2,
                max_size=10,
                command_timeout=30,
                server_settings={
                    "application_name": "provenance_service",
                    "jit": "off",
                },
            )

            # Test connection
            async with self.pg_pool.acquire() as conn:
                await conn.execute("SELECT 1")

            self.is_connected = True
            logger.info("Provenance service connected to database")
            return True

        except Exception as e:
            logger.error(f"Failed to connect to database: {str(e)}", error=str(e))
            return False

    async def disconnect(self) -> None:
        """Disconnect from database."""
        try:
            if self.pg_pool:
                await self.pg_pool.close()
                self.pg_pool = None
            self.is_connected = False
            logger.info("Provenance service disconnected")
        except Exception as e:
            logger.error(f"Error during disconnect: {str(e)}", error=str(e))

    async def ensure_schema(self) -> None:
        """
        Ensure database schema exists.

        Runs the SQL migration if tables don't exist.
        """
        try:
            schema_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                "..",
                "db",
                "provenance_schema.sql",
            )
            # Resolve absolute path
            schema_path = os.path.abspath(schema_path)

            if not os.path.exists(schema_path):
                logger.warning(f"Schema file not found at {schema_path}")
                logger.info("Creating tables directly via SQL commands instead")
                await self._create_tables_directly()
                return

            with open(schema_path, "r") as f:
                schema_sql = f.read()

            async with self.pg_pool.acquire() as conn:
                # Split by semicolons and execute each statement
                statements = [
                    s.strip()
                    for s in schema_sql.split(";")
                    if s.strip() and not s.strip().startswith("--")
                ]
                for statement in statements:
                    if statement:
                        try:
                            await conn.execute(statement)
                        except Exception as e:
                            # Ignore "already exists" errors
                            if "already exists" not in str(e).lower():
                                logger.warning(f"Schema statement warning: {str(e)}")

            logger.info("Provenance schema ensured")

        except Exception as e:
            logger.error(f"Failed to ensure schema: {str(e)}", error=str(e))
            raise

    async def _create_tables_directly(self) -> None:
        """Create tables directly if schema file not found."""
        async with self.pg_pool.acquire() as conn:
            # Create dataset_provenance table
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS dataset_provenance (
                    provenance_id VARCHAR(36) PRIMARY KEY,
                    dataset_id VARCHAR(100) NOT NULL UNIQUE,
                    dataset_name VARCHAR(255) NOT NULL,
                    source_info JSONB NOT NULL,
                    license_info JSONB NOT NULL,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    acquired_at TIMESTAMPTZ,
                    processed_at TIMESTAMPTZ,
                    validated_at TIMESTAMPTZ,
                    published_at TIMESTAMPTZ,
                    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    processing_lineage JSONB NOT NULL,
                    storage_info JSONB NOT NULL,
                    audit_info JSONB DEFAULT '{}'::jsonb,
                    metadata JSONB DEFAULT '{}'::jsonb,
                    provenance_document JSONB NOT NULL
                )
                """
            )

            # Create provenance_audit_log table
            await conn.execute(
                """
                CREATE TABLE IF NOT EXISTS provenance_audit_log (
                    audit_id VARCHAR(36) PRIMARY KEY,
                    provenance_id VARCHAR(36) NOT NULL,
                    dataset_id VARCHAR(100) NOT NULL,
                    action VARCHAR(50) NOT NULL,
                    changed_by VARCHAR(100) NOT NULL,
                    changed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                    old_value JSONB,
                    new_value JSONB,
                    change_reason TEXT
                )
                """
            )

            # Create basic indexes
            await conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_provenance_dataset_id ON dataset_provenance(dataset_id)"
            )

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
    )
    async def create_provenance(
        self,
        provenance: ProvenanceRecord,
        changed_by: str = "system",
    ) -> str:
        """
        Create a new provenance record.

        Args:
            provenance: ProvenanceRecord to create
            changed_by: User/system creating the record

        Returns:
            Provenance ID
        """
        if not self.is_connected:
            await self.connect()
            await self.ensure_schema()

        try:
            # Convert to dict
            prov_dict = provenance.to_dict()

            async with self.pg_pool.acquire() as conn:
                # Insert provenance record
                await conn.execute(
                    """
                    INSERT INTO dataset_provenance (
                        provenance_id, dataset_id, dataset_name,
                        source_info, license_info,
                        created_at, acquired_at, processed_at,
                        validated_at, published_at, updated_at,
                        processing_lineage, storage_info,
                        audit_info, metadata, provenance_document
                    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                    ON CONFLICT (provenance_id) DO UPDATE SET
                        dataset_name = EXCLUDED.dataset_name,
                        source_info = EXCLUDED.source_info,
                        license_info = EXCLUDED.license_info,
                        acquired_at = EXCLUDED.acquired_at,
                        processed_at = EXCLUDED.processed_at,
                        validated_at = EXCLUDED.validated_at,
                        published_at = EXCLUDED.published_at,
                        updated_at = NOW(),
                        processing_lineage = EXCLUDED.processing_lineage,
                        storage_info = EXCLUDED.storage_info,
                        audit_info = EXCLUDED.audit_info,
                        metadata = EXCLUDED.metadata,
                        provenance_document = EXCLUDED.provenance_document
                    """,
                    provenance.provenance_id,
                    provenance.dataset_id,
                    provenance.dataset_name,
                    json.dumps(prov_dict["source"]),
                    json.dumps(prov_dict["license"]),
                    provenance.timestamps.created_at,
                    provenance.timestamps.acquired_at,
                    provenance.timestamps.processed_at,
                    provenance.timestamps.validated_at,
                    provenance.timestamps.published_at,
                    provenance.timestamps.updated_at,
                    json.dumps(prov_dict["processing_lineage"]),
                    json.dumps(prov_dict["storage"]),
                    json.dumps(prov_dict["audit"]),
                    json.dumps(prov_dict["metadata"]),
                    json.dumps(prov_dict),
                )

                # Create audit log entry
                await self._create_audit_entry(
                    conn,
                    provenance.provenance_id,
                    provenance.dataset_id,
                    "created",
                    changed_by,
                    None,
                    prov_dict,
                )

            logger.info(
                f"Created provenance record: {provenance.provenance_id}",
                dataset_id=provenance.dataset_id,
            )

            # Store to S3
            await self._store_to_s3(provenance)

            return provenance.provenance_id

        except Exception as e:
            logger.error(
                f"Failed to create provenance: {str(e)}",
                error=str(e),
                dataset_id=provenance.dataset_id,
            )
            raise

    async def get_provenance(self, dataset_id: str) -> Optional[ProvenanceRecord]:
        """
        Get provenance record by dataset_id.

        Args:
            dataset_id: Dataset identifier

        Returns:
            ProvenanceRecord or None if not found
        """
        if not self.is_connected:
            await self.connect()

        try:
            async with self.pg_pool.acquire() as conn:
                row = await conn.fetchrow(
                    "SELECT provenance_document FROM dataset_provenance WHERE dataset_id = $1",
                    dataset_id,
                )

                if not row:
                    return None

                prov_data = row["provenance_document"]
                return ProvenanceRecord.from_dict(prov_data)

        except Exception as e:
            logger.error(
                f"Failed to get provenance: {str(e)}",
                error=str(e),
                dataset_id=dataset_id,
            )
            raise

    async def update_provenance(
        self,
        provenance: ProvenanceRecord,
        changed_by: str = "system",
        change_reason: Optional[str] = None,
    ) -> bool:
        """
        Update existing provenance record.

        Args:
            provenance: Updated ProvenanceRecord
            changed_by: User/system making the change
            change_reason: Reason for the update

        Returns:
            True if successful
        """
        if not self.is_connected:
            await self.connect()

        try:
            # Get old value for audit log
            old_provenance = await self.get_provenance(provenance.dataset_id)
            old_value = old_provenance.to_dict() if old_provenance else None

            # Convert to dict
            new_value = provenance.to_dict()
            prov_dict = provenance.to_dict()

            async with self.pg_pool.acquire() as conn:
                # Update provenance record
                await conn.execute(
                    """
                    UPDATE dataset_provenance SET
                        dataset_name = $1,
                        source_info = $2,
                        license_info = $3,
                        acquired_at = $4,
                        processed_at = $5,
                        validated_at = $6,
                        published_at = $7,
                        updated_at = NOW(),
                        processing_lineage = $8,
                        storage_info = $9,
                        audit_info = $10,
                        metadata = $11,
                        provenance_document = $12
                    WHERE dataset_id = $13
                    """,
                    provenance.dataset_name,
                    json.dumps(prov_dict["source"]),
                    json.dumps(prov_dict["license"]),
                    provenance.timestamps.acquired_at,
                    provenance.timestamps.processed_at,
                    provenance.timestamps.validated_at,
                    provenance.timestamps.published_at,
                    json.dumps(prov_dict["processing_lineage"]),
                    json.dumps(prov_dict["storage"]),
                    json.dumps(prov_dict["audit"]),
                    json.dumps(prov_dict["metadata"]),
                    json.dumps(prov_dict),
                    provenance.dataset_id,
                )

                # Create audit log entry
                await self._create_audit_entry(
                    conn,
                    provenance.provenance_id,
                    provenance.dataset_id,
                    "updated",
                    changed_by,
                    old_value,
                    new_value,
                    change_reason,
                )

            logger.info(
                f"Updated provenance record: {provenance.provenance_id}",
                dataset_id=provenance.dataset_id,
            )

            # Store updated version to S3
            await self._store_to_s3(provenance)

            return True

        except Exception as e:
            logger.error(
                f"Failed to update provenance: {str(e)}",
                error=str(e),
                dataset_id=provenance.dataset_id,
            )
            raise

    async def _create_audit_entry(
        self,
        conn: asyncpg.Connection,
        provenance_id: str,
        dataset_id: str,
        action: str,
        changed_by: str,
        old_value: Optional[Dict[str, Any]] = None,
        new_value: Optional[Dict[str, Any]] = None,
        change_reason: Optional[str] = None,
    ) -> str:
        """Create audit log entry."""
        audit_id = str(uuid4())

        await conn.execute(
            """
            INSERT INTO provenance_audit_log (
                audit_id, provenance_id, dataset_id, action,
                changed_by, old_value, new_value, change_reason
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            audit_id,
            provenance_id,
            dataset_id,
            action,
            changed_by,
            json.dumps(old_value) if old_value else None,
            json.dumps(new_value) if new_value else None,
            change_reason,
        )

        return audit_id

    async def _store_to_s3(self, provenance: ProvenanceRecord) -> None:
        """
        Store provenance document to S3.

        Args:
            provenance: ProvenanceRecord to store
        """
        try:
            # Lazy import boto3
            try:
                import boto3
            except ImportError:
                logger.warning("boto3 not available, skipping S3 storage")
                return

            if not self._s3_client:
                self._s3_client = boto3.client(
                    "s3",
                    region_name=self.s3_region,
                    endpoint_url=self.s3_endpoint_url,
                    aws_access_key_id=os.getenv("OVH_S3_ACCESS_KEY", os.getenv("AWS_ACCESS_KEY_ID")),
                    aws_secret_access_key=os.getenv("OVH_S3_SECRET_KEY", os.getenv("AWS_SECRET_ACCESS_KEY")),
                )

            # S3 key: provenance/{dataset_id}/v{version}/provenance.json
            # For now, use timestamp as version
            version = (
                provenance.timestamps.updated_at or provenance.timestamps.created_at
            ).strftime("%Y%m%d_%H%M%S")
            s3_key = f"provenance/{provenance.dataset_id}/v{version}/provenance.json"

            # Upload to S3
            self._s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=s3_key,
                Body=json.dumps(provenance.to_dict(), indent=2),
                ContentType="application/json",
            )

            logger.info(
                f"Stored provenance to S3: {s3_key}",
                dataset_id=provenance.dataset_id,
            )

        except Exception as e:
            # Don't fail the operation if S3 fails
            logger.warning(
                f"Failed to store provenance to S3: {str(e)}",
                error=str(e),
                dataset_id=provenance.dataset_id,
            )

    async def query_provenance(
        self,
        source_id: Optional[str] = None,
        license_type: Optional[str] = None,
        quality_tier: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[ProvenanceRecord]:
        """
        Query provenance records with filters.

        Args:
            source_id: Filter by source ID
            license_type: Filter by license type
            quality_tier: Filter by quality tier
            limit: Maximum results
            offset: Offset for pagination

        Returns:
            List of ProvenanceRecords
        """
        if not self.is_connected:
            await self.connect()

        try:
            conditions = []
            params = []
            param_num = 1

            if source_id:
                conditions.append(f"source_info->>'source_id' = ${param_num}")
                params.append(source_id)
                param_num += 1

            if license_type:
                conditions.append(f"license_info->>'license_type' = ${param_num}")
                params.append(license_type)
                param_num += 1

            if quality_tier:
                conditions.append(f"metadata->>'quality_tier' = ${param_num}")
                params.append(quality_tier)
                param_num += 1

            where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""

            params.extend([limit, offset])

            query = f"""
                SELECT provenance_document
                FROM dataset_provenance
                {where_clause}
                ORDER BY created_at DESC
                LIMIT ${param_num} OFFSET ${param_num + 1}
            """

            async with self.pg_pool.acquire() as conn:
                rows = await conn.fetch(query, *params)

            results = [
                ProvenanceRecord.from_dict(row["provenance_document"]) for row in rows
            ]

            logger.info(
                f"Queried {len(results)} provenance records",
                filters={
                    "source_id": source_id,
                    "license_type": license_type,
                    "quality_tier": quality_tier,
                },
            )

            return results

        except Exception as e:
            logger.error(f"Failed to query provenance: {str(e)}", error=str(e))
            raise

    async def get_audit_log(
        self,
        dataset_id: Optional[str] = None,
        provenance_id: Optional[str] = None,
        action: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get audit log entries.

        Args:
            dataset_id: Filter by dataset ID
            provenance_id: Filter by provenance ID
            action: Filter by action type
            limit: Maximum results

        Returns:
            List of audit log entries
        """
        if not self.is_connected:
            await self.connect()

        try:
            conditions = []
            params = []
            param_num = 1

            if dataset_id:
                conditions.append(f"dataset_id = ${param_num}")
                params.append(dataset_id)
                param_num += 1

            if provenance_id:
                conditions.append(f"provenance_id = ${param_num}")
                params.append(provenance_id)
                param_num += 1

            if action:
                conditions.append(f"action = ${param_num}")
                params.append(action)
                param_num += 1

            where_clause = f"WHERE {' AND '.join(conditions)}" if conditions else ""
            params.append(limit)

            query = f"""
                SELECT * FROM provenance_audit_log
                {where_clause}
                ORDER BY changed_at DESC
                LIMIT ${param_num}
            """

            async with self.pg_pool.acquire() as conn:
                rows = await conn.fetch(query, *params)

            results = [dict(row) for row in rows]
            return results

        except Exception as e:
            logger.error(f"Failed to get audit log: {str(e)}", error=str(e))
            raise

    @staticmethod
    def calculate_checksum(file_path: str | Path) -> str:
        """
        Calculate SHA256 checksum of a file.

        Args:
            file_path: Path to file

        Returns:
            SHA256 checksum as hex string
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)

        return sha256_hash.hexdigest()


# Global service instance (lazy initialization)
_service_instance: Optional[ProvenanceService] = None


async def get_provenance_service() -> ProvenanceService:
    """
    Get or create global ProvenanceService instance.

    Returns:
        ProvenanceService instance
    """
    global _service_instance
    if _service_instance is None:
        _service_instance = ProvenanceService()
        await _service_instance.connect()
        await _service_instance.ensure_schema()
    return _service_instance
