from __future__ import annotations

import sqlite3


class LocalForesightSchemaManager:
    @staticmethod
    def ensure_schema(conn: sqlite3.Connection) -> None:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS documents (
                bank_id TEXT NOT NULL,
                id TEXT NOT NULL,
                user_id TEXT,
                content TEXT NOT NULL,
                context TEXT,
                tags_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (bank_id, id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS document_tags (
                bank_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                tag TEXT NOT NULL,
                PRIMARY KEY (bank_id, document_id, tag)
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_documents_bank_user ON documents(bank_id, user_id, updated_at DESC)"
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_document_tags_lookup ON document_tags(bank_id, tag, document_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_document_tags_bank_tag ON document_tags(bank_id, tag)")
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_document_tags_category_lookup
            ON document_tags(bank_id, document_id, tag)
            WHERE tag GLOB 'category:*'
            """
        )
        conn.execute(
            """
            CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts
            USING fts5(
                bank_id UNINDEXED,
                document_id UNINDEXED,
                content,
                tokenize='porter unicode61'
            )
            """
        )
        conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS documents_ai AFTER INSERT ON documents
            BEGIN
                INSERT INTO documents_fts(bank_id, document_id, content)
                VALUES (new.bank_id, new.id, new.content);
            END
            """
        )
        conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS documents_ad AFTER DELETE ON documents
            BEGIN
                DELETE FROM documents_fts
                WHERE bank_id = old.bank_id AND document_id = old.id;
            END
            """
        )
        conn.execute(
            """
            CREATE TRIGGER IF NOT EXISTS documents_au AFTER UPDATE ON documents
            BEGIN
                UPDATE documents_fts
                SET content = new.content
                WHERE bank_id = old.bank_id AND document_id = old.id;
            END
            """
        )

        columns = {row[1] for row in conn.execute("PRAGMA table_info(documents)").fetchall()}
        if "user_id" not in columns:
            conn.execute("ALTER TABLE documents ADD COLUMN user_id TEXT")
