from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from core.tools import rag_tools
from core.tools import starter_dataset


def sanitize_filename_fragment(text: str, max_len: int = 20) -> str:
    preview = text[:max_len].strip()
    preview = re.sub(r"\s+", "_", preview)
    preview = re.sub(r"[^A-Za-z0-9_\-]+", "", preview)
    preview = preview.strip("_-")
    return preview or "starter"


def sample_starters_with_abs_ids(sample_size: int, seed: int) -> list[tuple[int, int, str]]:
    sampled = starter_dataset.sample_absolute_id_starters(sample_size, seed)
    return [(rel_id, abs_id, text) for rel_id, (abs_id, text) in enumerate(sampled, start=1)]


def build_system_prompt_with_starter(base_prompt: str, starter: str) -> str:
    prompt = base_prompt
    placeholder_patterns = [
        "{Insert the dataset starter sentence here}",
        "{starter}",
        "{{starter}}",
    ]
    replaced = False
    for token in placeholder_patterns:
        if token in prompt:
            prompt = prompt.replace(token, starter)
            replaced = True
    if not replaced:
        prompt = f"{prompt.rstrip()}\n\n## Starter Topic\n{starter}\n"
    return prompt


def existing_doc_ids(docs_dir: Path) -> dict[int, Path]:
    out: dict[int, Path] = {}
    for p in sorted(docs_dir.glob("*.md")):
        m = re.match(r"^(\d+)-.+\.md$", p.name)
        if not m:
            continue
        out[int(m.group(1))] = p
    return out


def generate_dossiers(force: bool = False) -> None:
    if not config.OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is missing")

    docs_dir = PROJECT_ROOT / config.DOCS_DIR
    docs_dir.mkdir(parents=True, exist_ok=True)

    sampled = sample_starters_with_abs_ids(config.SAMPLE_SIZE, config.RANDOM_SEED)
    existing = existing_doc_ids(docs_dir)

    client_kwargs: dict[str, Any] = {"api_key": config.OPENAI_API_KEY}
    if config.OPENAI_BASE_URL:
        client_kwargs["base_url"] = config.OPENAI_BASE_URL
    client = OpenAI(**client_kwargs)

    doc_model = config.REMOTE_AGENT
    doc_system_prompt = config.DOC_AGENT_SYSTEM_PROMPT.strip()

    total = len(sampled)
    generated = 0
    skipped = 0

    for relative_id, absolute_id, starter in sampled:
        name_preview = sanitize_filename_fragment(starter, max_len=20)
        out_path = docs_dir / f"{absolute_id}-{name_preview}.md"

        existing_path = existing.get(absolute_id)
        if existing_path and not force:
            skipped += 1
            print(f"[{relative_id}/{total}] Skip existing starter_id={absolute_id}: {existing_path.name}")
            continue

        if force:
            for old in docs_dir.glob(f"{absolute_id}-*.md"):
                if old.exists():
                    old.unlink()

        user_prompt = build_system_prompt_with_starter(doc_system_prompt, starter)
        resp = client.chat.completions.create(
            model=doc_model,
            temperature=config.TEMPERATURE,
            messages=[{"role": "user", "content": user_prompt}],
        )
        content = (resp.choices[0].message.content or "").strip()
        header = f"# Starter Dossier\n\n- topic: {starter}\n\n"
        out_path.write_text(header + content + "\n", encoding="utf-8")
        generated += 1
        print(f"[{relative_id}/{total}] Wrote: {out_path.name}")

    print(f"[generate] done: generated={generated}, skipped={skipped}, total_sampled={total}")


def parse_starter_id(md_path: Path) -> int:
    m = re.match(r"^(\d+)-.+\.md$", md_path.name)
    if not m:
        raise ValueError(f"filename does not start with starter id: {md_path.name}")
    return int(m.group(1))


def clean_chunk(text: str) -> str:
    return text.strip()


def require_index_imports() -> tuple[Any, Any, Any, Any]:
    try:
        from langchain_text_splitters import MarkdownTextSplitter
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Missing dependency: langchain-text-splitters") from exc

    try:
        from langchain_core.documents import Document
    except Exception:
        try:
            from langchain.schema import Document  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("Missing dependency: langchain-core (Document)") from exc

    try:
        from langchain_chroma import Chroma
    except Exception:
        try:
            from langchain_community.vectorstores import Chroma  # type: ignore
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError("Missing dependency: langchain-chroma or langchain-community") from exc

    try:
        import chromadb
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("Missing dependency: chromadb") from exc

    return MarkdownTextSplitter, Document, Chroma, chromadb


def build_index(
    *,
    docs_dir: Path,
    index_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    embedding_model: str,
    embedding_backend: str,
    device: str,
    reset: bool,
) -> None:
    MarkdownTextSplitter, Document, Chroma, chromadb = require_index_imports()

    docs_dir = docs_dir.resolve()
    index_dir = index_dir.resolve()
    chroma_dir = index_dir / "chroma_db"
    manifest_path = index_dir / "build_manifest.json"

    if not docs_dir.exists():
        raise FileNotFoundError(f"docs directory not found: {docs_dir}")
    index_dir.mkdir(parents=True, exist_ok=True)
    chroma_dir.mkdir(parents=True, exist_ok=True)

    md_files = sorted(docs_dir.glob("*.md"))
    if not md_files:
        raise FileNotFoundError(f"no markdown docs found in {docs_dir}")

    splitter = MarkdownTextSplitter(
        chunk_size=max(1, int(chunk_size)),
        chunk_overlap=max(0, int(chunk_overlap)),
    )

    if embedding_backend == "remote":
        embeddings = rag_tools.create_remote_embeddings(
            model_name=embedding_model,
            api_key=config.OPENAI_API_KEY,
            base_url=config.OPENAI_BASE_URL,
        )
    else:
        embeddings = rag_tools.create_local_embeddings(model_name=embedding_model, device=device)

    client = chromadb.PersistentClient(path=str(chroma_dir))

    grouped_docs: dict[int, list[Any]] = defaultdict(list)
    grouped_ids: dict[int, list[str]] = defaultdict(list)
    file_count_by_starter: dict[int, int] = defaultdict(int)

    for md_file in md_files:
        try:
            starter_id = parse_starter_id(md_file)
        except ValueError as exc:
            print(f"[WARN] skip file: {exc}")
            continue

        raw_text = md_file.read_text(encoding="utf-8")
        chunks = [clean_chunk(x) for x in splitter.split_text(raw_text)]
        chunks = [x for x in chunks if x]
        if not chunks:
            print(f"[WARN] no chunks produced for {md_file.name}")
            continue

        file_count_by_starter[starter_id] += 1
        for idx, chunk in enumerate(chunks, start=1):
            chunk_id = f"{md_file.stem}-c{idx:04d}"
            grouped_docs[starter_id].append(
                Document(
                    page_content=chunk,
                    metadata={
                        "starter_id": starter_id,
                        "source_file": md_file.name,
                        "chunk_id": chunk_id,
                    },
                )
            )
            grouped_ids[starter_id].append(chunk_id)

    if not grouped_docs:
        raise RuntimeError("no valid docs to index after parsing")

    starter_ids = sorted(grouped_docs.keys())
    total_chunks = 0

    for starter_id in starter_ids:
        collection_name = f"starter_{starter_id}"
        if reset:
            try:
                client.delete_collection(collection_name)
                print(f"[reset] deleted collection {collection_name}")
            except Exception:
                pass

        vs = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=str(chroma_dir),
            client=client,
        )
        docs = grouped_docs[starter_id]
        ids = grouped_ids[starter_id]
        vs.add_documents(documents=docs, ids=ids)
        if hasattr(vs, "persist"):
            vs.persist()

        total_chunks += len(docs)
        print(
            f"[indexed] starter_id={starter_id} files={file_count_by_starter[starter_id]} "
            f"chunks={len(docs)} collection={collection_name}"
        )

    manifest = {
        "built_at_utc": datetime.now(timezone.utc).isoformat(),
        "docs_dir": str(docs_dir),
        "index_dir": str(index_dir),
        "chroma_dir": str(chroma_dir),
        "embedding_backend": embedding_backend,
        "embedding_model": embedding_model,
        "chunk_size": int(chunk_size),
        "chunk_overlap": int(chunk_overlap),
        "device": device,
        "reset": bool(reset),
        "starter_count": len(starter_ids),
        "total_chunks": total_chunks,
        "starters": starter_ids,
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] manifest saved to {manifest_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare RAG corpus: generate starter dossiers and/or build index.")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser("generate", help="Generate missing starter dossier docs.")
    gen.add_argument("--force", action="store_true", help="Regenerate sampled docs even if they already exist.")

    idx = sub.add_parser("index", help="Build Chroma index from starter docs.")
    idx.add_argument("--docs-dir", type=Path, default=PROJECT_ROOT / config.DOCS_DIR)
    idx.add_argument("--index-dir", type=Path, default=PROJECT_ROOT / config.INDEX_DIR)
    idx.add_argument("--chunk-size", type=int, default=1200)
    idx.add_argument("--chunk-overlap", type=int, default=200)
    idx.add_argument("--embedding-model", type=str, default="BAAI/bge-large-en-v1.5")
    idx.add_argument("--embedding-backend", choices=("remote", "local"), default="remote")
    idx.add_argument("--device", type=str, default="cpu")
    idx.add_argument("--reset", action="store_true", help="Delete existing starter collections before indexing.")

    all_cmd = sub.add_parser("all", help="Generate missing docs then build index.")
    all_cmd.add_argument("--force", action="store_true", help="Regenerate sampled docs even if they already exist.")
    all_cmd.add_argument("--docs-dir", type=Path, default=PROJECT_ROOT / config.DOCS_DIR)
    all_cmd.add_argument("--index-dir", type=Path, default=PROJECT_ROOT / config.INDEX_DIR)
    all_cmd.add_argument("--chunk-size", type=int, default=1200)
    all_cmd.add_argument("--chunk-overlap", type=int, default=200)
    all_cmd.add_argument("--embedding-model", type=str, default="BAAI/bge-large-en-v1.5")
    all_cmd.add_argument("--embedding-backend", choices=("remote", "local"), default="remote")
    all_cmd.add_argument("--device", type=str, default="cpu")
    all_cmd.add_argument("--reset", action="store_true")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "generate":
        generate_dossiers(force=bool(args.force))
        return

    if args.command == "index":
        build_index(
            docs_dir=args.docs_dir,
            index_dir=args.index_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            embedding_model=args.embedding_model,
            embedding_backend=args.embedding_backend,
            device=args.device,
            reset=bool(args.reset),
        )
        return

    if args.command == "all":
        generate_dossiers(force=bool(args.force))
        build_index(
            docs_dir=args.docs_dir,
            index_dir=args.index_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
            embedding_model=args.embedding_model,
            embedding_backend=args.embedding_backend,
            device=args.device,
            reset=bool(args.reset),
        )
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
