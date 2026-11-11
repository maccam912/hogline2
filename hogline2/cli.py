import argparse
import shutil
import uuid
from pathlib import Path

from .config import Config
from .media import fingerprint, probe, youtube_id
from .store import Store


def main():
    parser = argparse.ArgumentParser(description="Download curling recordings and split complete ends")
    parser.add_argument("--data", help="Independent data directory")
    commands = parser.add_subparsers(dest="command", required=True)
    serve = commands.add_parser("serve")
    serve.add_argument("--port", type=int, default=8772)
    local = commands.add_parser("import-local", help="Copy an existing video into this app")
    local.add_argument("path", type=Path)
    local.add_argument("--url")
    local.add_argument("--title", default="Imported recording")
    local.add_argument("--expected-ends", type=int)
    analyze = commands.add_parser("analyze")
    analyze.add_argument("id")
    export = commands.add_parser("export")
    export.add_argument("id")
    args = parser.parse_args()
    config = Config.load(args.data)
    if args.command == "serve":
        import uvicorn

        from .app import create_app

        uvicorn.run(create_app(config), host="127.0.0.1", port=args.port)
    elif args.command == "import-local":
        if not args.path.is_file():
            parser.error("Source video does not exist")
        identifier = youtube_id(args.url) if args.url else "local-" + uuid.uuid4().hex
        store = Store(config.data)
        if any(r["id"] == identifier for r in store.all()):
            parser.error("This recording is already imported")
        root = config.data / "recordings" / identifier
        root.mkdir(parents=True, exist_ok=True)
        target = root / ("source" + args.path.suffix)
        shutil.copy2(args.path, target)
        store.create(
            identifier,
            title=args.title,
            url=args.url,
            source_path=str(target.resolve()),
            source_sha256=fingerprint(target),
            media=probe(target),
            expected_ends=args.expected_ends,
        )
        print(identifier)
    else:
        from .pipeline import Worker

        store = Store(config.data)
        worker = Worker(config, store)
        # CLI processing owns the same lock as the web worker; never run two writers.
        import fcntl

        with (config.data / "worker.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            worker.run(args.id, "analyze" if args.command == "analyze" else "export")
        print(store.get(args.id)["stage"])


if __name__ == "__main__":
    main()
