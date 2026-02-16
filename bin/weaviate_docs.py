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
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_weaviate import WeaviateVectorStore

logger = logging.getLogger(__name__)

DEFAULT_SCHEMA = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "config", "weaviate_collections.yaml",
)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


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


def create_collection_from_schema(client, name: str, schema: dict) -> None:
    """Create a Weaviate collection using a schema definition from YAML."""
    col_def = schema.get(name)
    if not col_def:
        raise ValueError(
            f"No schema definition for '{name}' in config. "
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


def ensure_collection(client, name: str, schema_path: str) -> None:
    """Create the collection if it doesn't already exist."""
    existing = client.collections.list_all()
    if name in existing:
        return
    schema = load_schema(schema_path)
    create_collection_from_schema(client, name, schema)
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
    """Load text files, attach metadata, split into chunks."""
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
    name = args.collection
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
    name = args.collection
    schema_path = args.schema or DEFAULT_SCHEMA
    try:
        schema = load_schema(schema_path)
        create_collection_from_schema(client, name, schema)
        print(f"Created collection '{name}' from {schema_path}")
    except Exception as e:
        print(f"Error creating collection: {e}")


def cmd_load(client, wv, args):
    """Load documents into a collection (auto-creates from schema if needed)."""
    name = args.collection
    schema_path = args.schema or DEFAULT_SCHEMA
    paths = resolve_paths(args.path, args.pattern)
    if not paths:
        print(f"No files found matching: {args.path} (pattern: {args.pattern})")
        return

    ensure_collection(client, name, schema_path)

    print(f"Loading {len(paths)} file(s) into '{name}'...")
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


def cmd_update(client, wv, args):
    """Update documents: delete by source, then re-load."""
    name = args.collection
    paths = resolve_paths(args.path, args.pattern)
    if not paths:
        print(f"No files found matching: {args.path}")
        return

    col = client.collections.get(name)
    for fp in paths:
        source = os.path.basename(fp)
        deleted = _delete_by_source(col, source)
        print(f"  Deleted {deleted} existing chunk(s) for '{source}'")

    print(f"Re-loading {len(paths)} file(s)...")
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


def cmd_delete(client, wv, args):
    """Delete documents by source filename."""
    name = args.collection
    col = client.collections.get(name)
    for source in args.source:
        deleted = _delete_by_source(col, source)
        print(f"Deleted {deleted} chunk(s) with source='{source}' from '{name}'")


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
    name = args.collection
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
    name = args.collection
    if not args.yes:
        confirm = input(f"DROP collection '{name}' (schema + data)? [y/N] ")
        if confirm.lower() != "y":
            print("Cancelled.")
            return
    client.collections.delete(name)
    print(f"Dropped collection: {name}")


def cmd_search(client, wv, args):
    """Search a collection (for testing)."""
    name = args.collection
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


# ── CLI entry point ───────────────────────────────────


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

    # update
    p = sub.add_parser("update", help="Re-index documents (delete + re-load)")
    p.add_argument("collection")
    p.add_argument("path", help="File or directory to update")
    p.add_argument("--pattern", default="*.md", help="Glob pattern (default: *.md)")

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

    # search
    p = sub.add_parser("search", help="Search a collection")
    p.add_argument("collection")
    p.add_argument("query")
    p.add_argument("-k", type=int, default=4, help="Number of results (default: 4)")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    client, wv = connect()
    try:
        cmds = {
            "list": cmd_list, "info": cmd_info, "create": cmd_create,
            "load": cmd_load, "update": cmd_update, "delete": cmd_delete,
            "clear": cmd_clear, "drop": cmd_drop, "search": cmd_search,
        }
        cmds[args.command](client, wv, args)
    finally:
        client.close()


if __name__ == "__main__":
    main()
