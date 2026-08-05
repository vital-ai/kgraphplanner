#!/usr/bin/env python3
"""
CLI for managing documents and collections in Weaviate.

Subcommands:
    list        List all collections with object counts
    info        Show details for a collection
    create      Create an empty collection from schema
    load        Load documents from files/directories into a collection
    update      Re-index documents (delete + re-load by source)
    delete      Delete documents by source filename
    clear       Delete all objects in a collection (keep schema)
    drop        Drop a collection entirely (schema + data)
    search      Search a collection (for testing)

Reads connection config from KGPLAN__WEAVIATE__* env vars via AgentConfig.
JWT auth is obtained via WEAVIATE_KEYCLOAK_* env vars.

Usage:
    python bin/weaviate_docs.py <command> [args]
"""

import os
import sys
import re
import glob
import argparse
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import yaml
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType, Configure
from weaviate.config import AdditionalConfig, Timeout

from kgraphplanner.config.agent_config import AgentConfig
from kgraphplanner.weaviate.embeddings import get_embeddings

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from langchain_weaviate import WeaviateVectorStore
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

DEFAULT_SCHEMA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "weaviate_collections.yaml",
)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
MARKDOWN_MAX_CHUNK = 2000  # sections larger than this get a secondary char split


# ── Schema helpers ────────────────────────────────────

_DATA_TYPE_MAP = {
    "text": DataType.TEXT,
    "int": DataType.INT,
    "number": DataType.NUMBER,
    "bool": DataType.BOOL,
    "date": DataType.DATE,
    "text[]": DataType.TEXT_ARRAY,
}


def load_schema(schema_path: str) -> dict:
    """Load collection schemas from YAML file."""
    with open(schema_path, "r") as f:
        data = yaml.safe_load(f)
    return data.get("collections", {})


def create_collection_from_schema(client, name: str, schema: dict, schema_key: str = "") -> None:
    """Create a Weaviate collection using a schema definition from YAML.

    Args:
        name: The actual collection name in Weaviate (may include prefix).
        schema: Parsed schema dict.
        schema_key: Key to look up in schema (unprefixed). Defaults to name.
    """
    key = schema_key or name
    col_def = schema.get(key)
    if not col_def:
        raise ValueError(
            f"No schema definition for '{key}' in config. "
            f"Available: {list(schema.keys())}"
        )

    properties = []
    for prop in col_def.get("properties", []):
        dt = _DATA_TYPE_MAP.get(prop["data_type"], DataType.TEXT)
        kwargs = {"name": prop["name"], "data_type": dt}
        if "description" in prop:
            kwargs["description"] = prop["description"]
        if prop.get("indexFilterable"):
            kwargs["index_filterable"] = True
        properties.append(Property(**kwargs))

    vectorizer = col_def.get("vectorizer", "none")
    description = col_def.get("description", "")

    client.collections.create(
        name=name,
        description=description,
        vectorizer_config=Configure.Vectorizer.none() if vectorizer == "none" else None,
        properties=properties,
    )


def ensure_collection(client, name: str, schema_path: str, schema_key: str = "") -> None:
    """Create the collection if it doesn't already exist."""
    existing = client.collections.list_all()
    if name in existing:
        return
    schema = load_schema(schema_path)
    create_collection_from_schema(client, name, schema, schema_key=schema_key or name)
    print(f"  Auto-created collection '{name}' from schema")


# ── Connection ────────────────────────────────────────


def connect() -> tuple:
    """Connect to Weaviate using AgentConfig + Keycloak JWT.
    Returns (client, weaviate_config).
    """
    from kgraphplanner.weaviate.auth import get_weaviate_jwt

    config = AgentConfig.from_env()
    wv = config.weaviate

    token, err = get_weaviate_jwt()
    if err:
        logger.warning(f"JWT auth: {err}")

    connect_kwargs = dict(
        http_host=wv.http_host,
        http_port=wv.http_port,
        http_secure=wv.http_secure,
        grpc_host=wv.grpc_host or wv.http_host,
        grpc_port=wv.grpc_port,
        grpc_secure=wv.grpc_secure,
        skip_init_checks=wv.skip_init_checks,
        additional_config=AdditionalConfig(
            timeout=Timeout(init=10, query=30, insert=60)
        ),
    )
    if token and wv.auth_mode == "bearer":
        connect_kwargs["auth_credentials"] = Auth.bearer_token(token)

    client = weaviate.connect_to_custom(**connect_kwargs)
    return client, wv


# ── Prefix helper ─────────────────────────────────────


def prefixed_name(name: str, wv) -> str:
    """Apply the environment collection prefix (e.g. 'Dev' -> 'DevxxxKnowledgeBase')."""
    prefix = wv.collection_prefix
    if prefix and not name.startswith(prefix + "xxx"):
        return prefix + "xxx" + name
    return name


# ── Document helpers ──────────────────────────────────


def resolve_paths(path: str, pattern: str = "*.md") -> list[str]:
    """Resolve a file or directory + glob pattern to a list of file paths."""
    if os.path.isfile(path):
        return [path]
    if os.path.isdir(path):
        matches = sorted(glob.glob(os.path.join(path, "**", pattern), recursive=True))
        if not matches:
            matches = sorted(glob.glob(os.path.join(path, pattern)))
        return matches
    # Treat as glob
    return sorted(glob.glob(path))


def load_and_split(file_paths: list[str], collection_name: str) -> list:
    """Load text files, attach metadata, split into chunks (character-based)."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " "],
    )
    all_docs = []
    for fp in file_paths:
        loader = TextLoader(fp, encoding="utf-8")
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = os.path.basename(fp)
            doc.metadata["collection"] = collection_name
        chunks = splitter.split_documents(docs)
        all_docs.extend(chunks)
        print(f"  {os.path.basename(fp)}: {len(docs)} doc(s) -> {len(chunks)} chunks")
    return all_docs


def load_and_split_markdown(file_paths: list[str], collection_name: str) -> list:
    """Load Markdown files and split on logical header boundaries.

    Uses MarkdownHeaderTextSplitter to split at ## and ### boundaries,
    preserving the header hierarchy as metadata. Sections that exceed
    MARKDOWN_MAX_CHUNK chars get a secondary character-based split.
    """
    headers_to_split_on = [
        ("#", "header_1"),
        ("##", "header_2"),
        ("###", "header_3"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )
    secondary_splitter = RecursiveCharacterTextSplitter(
        chunk_size=MARKDOWN_MAX_CHUNK,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n#### ", "\n\n", "\n", " "],
    )

    all_docs = []
    for fp in file_paths:
        source = os.path.basename(fp)
        with open(fp, "r", encoding="utf-8") as f:
            text = f.read()

        md_sections = md_splitter.split_text(text)

        file_chunks = []
        for section in md_sections:
            # Build section title from header metadata
            section_title_parts = []
            for key in ["header_1", "header_2", "header_3"]:
                if key in section.metadata:
                    section_title_parts.append(section.metadata[key])
            section_title = " > ".join(section_title_parts)

            base_meta = {
                "source": source,
                "collection": collection_name,
                "section": section_title,
            }
            # Merge header metadata
            for key in ["header_1", "header_2", "header_3"]:
                if key in section.metadata:
                    base_meta[key] = section.metadata[key]

            content = section.page_content

            if len(content) <= MARKDOWN_MAX_CHUNK:
                doc = Document(page_content=content, metadata=base_meta.copy())
                file_chunks.append(doc)
            else:
                # Secondary split for oversized sections
                sub_chunks = secondary_splitter.split_text(content)
                for i, chunk_text in enumerate(sub_chunks):
                    meta = base_meta.copy()
                    meta["sub_chunk"] = i
                    doc = Document(page_content=chunk_text, metadata=meta)
                    file_chunks.append(doc)

        all_docs.extend(file_chunks)
        print(f"  {source}: {len(md_sections)} section(s) -> {len(file_chunks)} chunks (markdown)")

    return all_docs


# ── Dialog loading ────────────────────────────────────

VOICE_MAP = {
    "agent_carly": "carly",
    "agent_reed": "reed",
    "agent_morgan": "morgan",
}


def _parse_dialog_metadata(text: str) -> dict:
    """Extract structured metadata from a dialog markdown file."""
    meta = {}

    # Phase
    m = re.search(r"\*\*Phase:\*\*\s*(.+)", text)
    if m:
        meta["phase"] = m.group(1).strip()

    # Product
    m = re.search(r"\*\*Product:\*\*\s*(.+)", text)
    if m:
        meta["product"] = m.group(1).strip()

    # Industry
    m = re.search(r"\*\*Industry:\*\*\s*(.+)", text)
    if m:
        meta["industry"] = m.group(1).strip()

    # Personality
    m = re.search(r"\*\*Personality:\*\*\s*(.+)", text)
    if m:
        meta["personality"] = m.group(1).strip()

    # Scenario title from first H1
    m = re.search(r"^#\s+(.+)", text, re.MULTILINE)
    if m:
        meta["scenario"] = m.group(1).strip()

    return meta


def load_dialogs(file_paths: list[str], collection_name: str) -> list:
    """Load dialog files as single documents with voice and scenario metadata.

    Each dialog becomes one Document (not chunked further) since they are
    typically 3-6KB — suitable for single-embedding retrieval.
    Voice is derived from the parent folder name (agent_carly -> carly).
    """
    all_docs = []
    for fp in file_paths:
        source = os.path.basename(fp)
        parent_dir = os.path.basename(os.path.dirname(fp))
        voice = VOICE_MAP.get(parent_dir, parent_dir)

        with open(fp, "r", encoding="utf-8") as f:
            text = f.read()

        meta = _parse_dialog_metadata(text)
        meta["source"] = source
        meta["collection"] = collection_name
        meta["voice"] = voice

        doc = Document(page_content=text, metadata=meta)
        all_docs.append(doc)

    print(f"  {len(all_docs)} dialog(s) loaded (voice: {voice if all_docs else '?'})")
    return all_docs


# ── Source document storage ───────────────────────────

SOURCE_DOCS_COLLECTION = "SourceDocuments"


def store_source_documents(client, file_paths: list[str], collection_name: str, schema_path: str, weaviate_name: str = "") -> int:
    """Store full document text in SourceDocuments for retrieval by name.

    Each source file gets one object (upsert by source name).
    No embeddings are computed — this is a pure lookup collection.
    """
    actual_name = weaviate_name or SOURCE_DOCS_COLLECTION
    ensure_collection(client, actual_name, schema_path, schema_key=SOURCE_DOCS_COLLECTION)
    col = client.collections.get(actual_name)

    stored = 0
    for fp in file_paths:
        source = os.path.basename(fp)
        with open(fp, "r", encoding="utf-8") as f:
            full_text = f.read()

        # Count chunks for metadata
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP,
            separators=["\n## ", "\n### ", "\n#### ", "\n\n", "\n", " "],
        )
        chunk_count = len(splitter.split_text(full_text))

        # Delete existing entry for this source (upsert)
        _delete_by_source(col, source)

        # Insert full document
        col.data.insert({
            "text": full_text,
            "source": source,
            "path": fp,
            "collection": collection_name,
            "chunk_count": chunk_count,
        })
        stored += 1

    return stored


# ── Commands ──────────────────────────────────────────


def cmd_list(client, wv, args):
    """List all collections with object counts."""
    collections = client.collections.list_all()
    if not collections:
        print("No collections found.")
        return
    print(f"{'Collection':<30} {'Objects':>10}")
    print("-" * 42)
    for name in sorted(collections.keys()):
        try:
            col = client.collections.get(name)
            count = col.aggregate.over_all(total_count=True).total_count
        except Exception:
            count = "?"
        print(f"{name:<30} {count:>10}")


def cmd_info(client, wv, args):
    """Show details for a collection."""
    name = prefixed_name(args.collection, wv)
    try:
        col = client.collections.get(name)
        count = col.aggregate.over_all(total_count=True).total_count
        print(f"Collection: {name}")
        print(f"Objects:    {count}")
        # Show a sample of sources
        results = col.query.fetch_objects(limit=100)
        sources = set()
        for obj in results.objects:
            src = obj.properties.get("source", "")
            if src:
                sources.add(src)
        if sources:
            print(f"Sources ({len(sources)}):")
            for s in sorted(sources):
                print(f"  - {s}")
    except Exception as e:
        print(f"Error: {e}")


def cmd_create(client, wv, args):
    """Create an empty collection from YAML schema."""
    schema_key = args.collection
    name = prefixed_name(schema_key, wv)
    schema_path = args.schema or DEFAULT_SCHEMA
    try:
        schema = load_schema(schema_path)
        create_collection_from_schema(client, name, schema, schema_key=schema_key)
        print(f"Created collection '{name}' from {schema_path}")
    except Exception as e:
        print(f"Error creating collection: {e}")


def cmd_load(client, wv, args):
    """Load documents into a collection (auto-creates from schema if needed)."""
    schema_key = args.collection
    name = prefixed_name(schema_key, wv)
    schema_path = args.schema or DEFAULT_SCHEMA
    paths = resolve_paths(args.path, args.pattern)
    if not paths:
        print(f"No files found matching: {args.path} (pattern: {args.pattern})")
        return

    ensure_collection(client, name, schema_path, schema_key=schema_key)

    split_mode = getattr(args, "split", "chars")
    print(f"Loading {len(paths)} file(s) into '{name}' (split: {split_mode})...")

    if split_mode == "markdown":
        chunks = load_and_split_markdown(paths, name)
    else:
        chunks = load_and_split(paths, name)

    if not chunks:
        print("No chunks produced.")
        return

    embeddings = get_embeddings(wv)
    WeaviateVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=client,
        index_name=name,
        text_key="text",
    )
    print(f"Indexed {len(chunks)} chunks into '{name}'")

    # Store full documents for retrieval by name
    src_name = prefixed_name(SOURCE_DOCS_COLLECTION, wv)
    stored = store_source_documents(client, paths, name, schema_path, weaviate_name=src_name)
    print(f"Stored {stored} full document(s) in '{src_name}'")


def cmd_load_dialogs(client, wv, args):
    """Load dialog files into CustomerDialogs with voice and scenario metadata."""
    schema_key = "CustomerDialogs"
    collection_name = prefixed_name(schema_key, wv)
    schema_path = args.schema or DEFAULT_SCHEMA
    paths = resolve_paths(args.path, args.pattern)
    if not paths:
        print(f"No files found matching: {args.path} (pattern: {args.pattern})")
        return

    ensure_collection(client, collection_name, schema_path, schema_key=schema_key)

    print(f"Loading {len(paths)} dialog(s) into '{collection_name}'...")
    docs = load_dialogs(paths, collection_name)

    if not docs:
        print("No documents produced.")
        return

    embeddings = get_embeddings(wv)
    WeaviateVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        client=client,
        index_name=collection_name,
        text_key="text",
    )
    print(f"Indexed {len(docs)} dialog(s) into '{collection_name}'")


def cmd_update(client, wv, args):
    """Update documents: delete by source, then re-load."""
    name = prefixed_name(args.collection, wv)
    paths = resolve_paths(args.path, args.pattern)
    if not paths:
        print(f"No files found matching: {args.path}")
        return

    col = client.collections.get(name)
    for fp in paths:
        source = os.path.basename(fp)
        deleted = _delete_by_source(col, source)
        print(f"  Deleted {deleted} existing chunk(s) for '{source}'")

    split_mode = getattr(args, "split", "chars")
    print(f"Re-loading {len(paths)} file(s) (split: {split_mode})...")

    if split_mode == "markdown":
        chunks = load_and_split_markdown(paths, name)
    else:
        chunks = load_and_split(paths, name)

    if chunks:
        embeddings = get_embeddings(wv)
        WeaviateVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            client=client,
            index_name=name,
            text_key="text",
        )
    print(f"Updated {len(chunks)} chunks in '{name}'")

    # Update full documents in SourceDocuments
    schema_path = args.schema or DEFAULT_SCHEMA
    src_name = prefixed_name(SOURCE_DOCS_COLLECTION, wv)
    stored = store_source_documents(client, paths, name, schema_path, weaviate_name=src_name)
    print(f"Updated {stored} full document(s) in '{src_name}'")


def cmd_delete(client, wv, args):
    """Delete documents by source filename."""
    name = prefixed_name(args.collection, wv)
    col = client.collections.get(name)
    src_name = prefixed_name(SOURCE_DOCS_COLLECTION, wv)
    for source in args.source:
        deleted = _delete_by_source(col, source)
        print(f"Deleted {deleted} chunk(s) with source='{source}' from '{name}'")
        # Also remove from SourceDocuments
        try:
            src_col = client.collections.get(src_name)
            src_deleted = _delete_by_source(src_col, source)
            if src_deleted:
                print(f"  Also removed {src_deleted} entry from '{src_name}'")
        except Exception:
            pass  # SourceDocuments collection may not exist yet


def _delete_by_source(col, source: str) -> int:
    """Delete all objects in a collection where source == filename."""
    from weaviate.classes.query import Filter
    results = col.query.fetch_objects(
        filters=Filter.by_property("source").equal(source),
        limit=10000,
    )
    ids = [obj.uuid for obj in results.objects]
    for uid in ids:
        col.data.delete_by_id(uid)
    return len(ids)


def cmd_clear(client, wv, args):
    """Delete all objects in a collection (keep schema)."""
    name = prefixed_name(args.collection, wv)
    if not args.yes:
        confirm = input(f"Delete ALL objects in '{name}'? [y/N] ")
        if confirm.lower() != "y":
            print("Cancelled.")
            return
    col = client.collections.get(name)
    col.data.delete_many(where=None)
    print(f"Cleared all objects in '{name}'")


def cmd_drop(client, wv, args):
    """Drop a collection entirely (schema + data)."""
    name = prefixed_name(args.collection, wv)
    if not args.yes:
        confirm = input(f"DROP collection '{name}' (schema + data)? [y/N] ")
        if confirm.lower() != "y":
            print("Cancelled.")
            return
    client.collections.delete(name)
    print(f"Dropped collection: {name}")


def cmd_search(client, wv, args):
    """Search a collection (for testing)."""
    name = prefixed_name(args.collection, wv)
    query = args.query
    k = args.k

    embeddings = get_embeddings(wv)
    store = WeaviateVectorStore(
        client=client,
        index_name=name,
        text_key="text",
        embedding=embeddings,
    )
    docs = store.similarity_search(query, k=k)
    print(f"Search: '{query}' in '{name}' (top {k})\n")
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        print(f"--- Result {i} (source: {source}) ---")
        print(doc.page_content[:500])
        print()
    if not docs:
        print("No results found.")


def cmd_get(client, wv, args):
    """Retrieve a full document from SourceDocuments by source name."""
    source = args.source
    src_name = prefixed_name(SOURCE_DOCS_COLLECTION, wv)
    from weaviate.classes.query import Filter

    try:
        col = client.collections.get(src_name)
        results = col.query.fetch_objects(
            filters=Filter.by_property("source").equal(source),
            limit=1,
        )
        if not results.objects:
            print(f"No document found with source='{source}' in {src_name}")
            return

        obj = results.objects[0]
        props = obj.properties
        print(f"Source:      {props.get('source', '')}")
        print(f"Path:        {props.get('path', '')}")
        print(f"Collection:  {props.get('collection', '')}")
        print(f"Chunks:      {props.get('chunk_count', '?')}")
        print(f"Text length: {len(props.get('text', ''))} chars")
        print("\n--- Full text ---\n")
        print(props.get("text", ""))
    except Exception as e:
        print(f"Error: {e}")


# ── CLI entry point ───────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Weaviate document management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # list
    sub.add_parser("list", help="List all collections")

    # info
    p = sub.add_parser("info", help="Show collection details")
    p.add_argument("collection")

    # create
    p = sub.add_parser("create", help="Create an empty collection from schema")
    p.add_argument("collection")
    p.add_argument("--schema", default=None, help="Path to YAML schema (default: config/weaviate_collections.yaml)")

    # load
    p = sub.add_parser("load", help="Load documents into a collection")
    p.add_argument("collection")
    p.add_argument("path", help="File or directory to load")
    p.add_argument("--pattern", default="*.md", help="Glob pattern (default: *.md)")
    p.add_argument("--schema", default=None, help="Path to YAML schema (default: config/weaviate_collections.yaml)")
    p.add_argument("--split", choices=["chars", "markdown"], default="chars",
                   help="Split strategy: 'chars' (fixed-size) or 'markdown' (header-aware logical boundaries)")

    # update
    p = sub.add_parser("update", help="Re-index documents (delete + re-load)")
    p.add_argument("collection")
    p.add_argument("path", help="File or directory to update")
    p.add_argument("--pattern", default="*.md", help="Glob pattern (default: *.md)")
    p.add_argument("--split", choices=["chars", "markdown"], default="chars",
                   help="Split strategy: 'chars' (fixed-size) or 'markdown' (header-aware logical boundaries)")

    # delete
    p = sub.add_parser("delete", help="Delete documents by source filename")
    p.add_argument("collection")
    p.add_argument("--source", nargs="+", required=True, help="Source filename(s)")

    # clear
    p = sub.add_parser("clear", help="Delete all objects in a collection")
    p.add_argument("collection")
    p.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")

    # drop
    p = sub.add_parser("drop", help="Drop collection (schema + data)")
    p.add_argument("collection")
    p.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")

    # load-dialogs
    p = sub.add_parser("load-dialogs", help="Load dialog files into CustomerDialogs with voice metadata")
    p.add_argument("path", help="Directory containing dialog files (e.g. agent_carly/)")
    p.add_argument("--pattern", default="*.md", help="Glob pattern (default: *.md)")
    p.add_argument("--schema", default=None, help="Path to YAML schema")

    # search
    p = sub.add_parser("search", help="Search a collection")
    p.add_argument("collection")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=4, help="Number of results (default: 4)")

    # get
    p = sub.add_parser("get", help="Retrieve a full document by source name")
    p.add_argument("source", help="Source filename to retrieve")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    client, wv = connect()
    try:
        cmds = {
            "list": cmd_list, "info": cmd_info, "create": cmd_create,
            "load": cmd_load, "load-dialogs": cmd_load_dialogs,
            "update": cmd_update, "delete": cmd_delete,
            "clear": cmd_clear, "drop": cmd_drop, "search": cmd_search,
            "get": cmd_get,
        }
        cmds[args.command](client, wv, args)
    finally:
        client.close()


if __name__ == "__main__":
    main()
