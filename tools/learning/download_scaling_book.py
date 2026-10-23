#!/usr/bin/env python3
"""Fetch a pinned local reading bundle. Standard library only; no upstream code runs."""

import argparse
import hashlib
import html
import json
from pathlib import Path, PurePosixPath
import re
import tarfile
import tempfile
from urllib.parse import unquote, urljoin, urlsplit
from urllib.request import Request, urlopen


MANIFEST = Path(__file__).with_name("scaling_book_manifest.json")
SITE = "https://jax-ml.github.io/scaling-book/"
MAX_ARCHIVE = 80 * 1024 * 1024
MAX_EXPANDED = 100 * 1024 * 1024
MAX_FILE = 8 * 1024 * 1024
FENCES = re.compile(r"(?ms)^```[^\n]*\n.*?^```[ \t]*$")
INLINE_CODE = re.compile(r"(`+).*?\1", re.S)
MD_LINK = re.compile(r"(?P<before>\]\()(?P<url>[^\s)]+)(?P<after>\))")
HTML_LINK = re.compile(r"(?P<before>\b(?:href|src)=[\"'])(?P<url>[^\"']+)(?P<after>[\"'])")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def safe_path(name):
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or not path.parts:
        raise ValueError(f"Unsafe archive/manifest path: {name}")
    return path


def outside_git(path):
    if any((p / ".git").exists() for p in (path, *path.parents)):
        raise ValueError("Choose an output directory outside every Git checkout")


def collect_archive(archive, manifest):
    """Read only regular allowlisted files; never extract archive filesystem entries."""
    if archive.stat().st_size > MAX_ARCHIVE:
        raise ValueError("Archive exceeds size limit")
    expected = {entry["path"]: entry for entry in manifest["files"]}
    if len(expected) != len(manifest["files"]):
        raise ValueError("Duplicate manifest path")
    for name in expected:
        safe_path(name)
        if not (name in {"LICENSE", "README.md"} or name.endswith(".md")
                or (name.startswith("assets/") and Path(name).suffix in {".png", ".gif"})):
            raise ValueError(f"Unsupported selected file: {name}")
    found, names, roots, total = {}, set(), set(), 0
    with tarfile.open(archive, "r:gz") as tar:
        for number, member in enumerate(tar, 1):
            if number > 1500:
                raise ValueError("Too many archive entries")
            path = safe_path(member.name)
            roots.add(path.parts[0])
            if len(roots) != 1 or member.name in names:
                raise ValueError("Multiple roots or duplicate archive path")
            names.add(member.name)
            if member.isdir():
                continue
            if not member.isfile():
                raise ValueError(f"Archive links/special entries are forbidden: {member.name}")
            total += member.size
            if member.size > MAX_FILE or total > MAX_EXPANDED:
                raise ValueError("Expanded archive exceeds size limit")
            name = str(PurePosixPath(*path.parts[1:]))
            if name not in expected:
                continue
            entry = expected[name]
            if member.size != entry["bytes"]:
                raise ValueError(f"Size mismatch: {name}")
            data = tar.extractfile(member).read(MAX_FILE + 1)
            if sha256(data) != entry["sha256"]:
                raise ValueError(f"SHA-256 mismatch: {name}")
            found[name] = data
    if set(found) != set(expected):
        raise ValueError(f"Missing selected files: {sorted(set(expected) - set(found))}")
    return found


def rewrite_links(text, transform):
    # Leave examples untouched: the download never executes or edits their code.
    snippets = []

    def stash(match):
        snippets.append(match.group())
        return f"OFFLINE_CODE_BLOCK_{len(snippets) - 1}"

    text = INLINE_CODE.sub(stash, FENCES.sub(stash, text))
    for pattern in (MD_LINK, HTML_LINK):
        text = pattern.sub(lambda m: m["before"] + transform(m["url"]) + m["after"], text)
    for index, snippet in enumerate(snippets):
        text = text.replace(f"OFFLINE_CODE_BLOCK_{index}", snippet)
    return text


def anchors(text):
    ids = set(re.findall(r"\b(?:id|name)=[\"']([^\"']+)", text))
    seen = {}
    for heading in re.findall(r"(?m)^#{1,6}\s+(.+)$", FENCES.sub("", text)):
        heading = re.sub(r"<[^>]*>", "", heading)
        heading = re.sub(r"[^\w\- ]", "", html.unescape(heading)).strip().lower().replace(" ", "-")
        count = seen.get(heading, 0)
        ids.add(heading + (f"-{count}" if count else ""))
        seen[heading] = count + 1
    return ids


def render_markdown(chapter, source, manifest):
    name, title = chapter["path"], chapter["title"]
    page = SITE if name == "index.md" else SITE + Path(name).stem + "/"
    text = source.decode("utf-8")
    text = re.sub(r"\A---\s*\n.*?\n---\s*\n", "", text, count=1, flags=re.S)
    chapters = {Path(c["path"]).stem for c in manifest["chapters"]}

    def figure(match):
        attrs = dict(re.findall(r'(\w+)="((?:\\.|[^"\\])*)"', match[1], re.S))
        path = attrs["path"]
        if path not in {entry["path"] for entry in manifest["files"]}:
            raise ValueError(f"Figure absent from manifest: {path}")
        caption = attrs.get("caption", "").replace('\\"', '"').replace("\\'", "'")
        return f"\n\n![Figure: {Path(path).name}]({path})\n\n{caption}\n\n"

    text = re.sub(r"{%\s*include\s+figure\.liquid\s+(.*?)%}", figure, text, flags=re.S)
    text = re.sub(r"{%\s*details\s+(.*?)\s*%}", r"\n\n**Answer/details: \1**\n\n", text, flags=re.S)
    text = re.sub(r"{%\s*(?:enddetails|raw|endraw)\s*%}", "", text)

    def plot(match):
        asset = re.search(r"assets/plotly/[\w-]+\.html", match[0])
        if not asset:
            raise ValueError("Unrecognized iframe; review before updating")
        return f"\n\n[Interactive plot (online; omitted offline)]({SITE}{asset[0]})\n\n"

    text = re.sub(r"<iframe\b.*?</iframe>", plot, text, flags=re.S)
    text = re.sub(r'<d-cite\s+key="([^"]+)"\s*></d-cite>',
                  lambda m: f" [citation {m[1]} (online)]({page})", text)
    text = text.replace("<d-footnote>", " (Footnote: ").replace("</d-footnote>", ") ")
    if re.search(r"{%|{{|<script\b|<iframe\b", INLINE_CODE.sub("", FENCES.sub("", text))):
        raise ValueError(f"Unprocessed template or active content in {name}")

    def localize(url):
        parts = urlsplit(html.unescape(url))
        if parts.scheme or parts.netloc or not parts.path:
            return url
        path = parts.path.removeprefix("../").removeprefix("./").strip("/")
        path = path.removeprefix("scaling-book/")
        suffix = ("#" + parts.fragment) if parts.fragment else ""
        if path.startswith("assets/"):
            return "../source/" + path + suffix
        if path.removesuffix(".md") in chapters:
            return path.removesuffix(".md") + ".md" + suffix
        return urljoin(page, url)

    text = rewrite_links(text, localize)
    return (f"# {title}\n\n"
            f"> Derived locally from upstream `{manifest['commit']}` for source reading. "
            f"[Exact source](../source/{name}) · [Upstream page]({page}) · "
            "[License](../source/LICENSE) · [Attribution](../source/README.md). "
            "TeX remains source; interactive plots and citation resolution need the website.\n\n" + text)


def check_links(root, documents):
    checked, online = 0, 0
    for relative, text in documents.items():
        text = INLINE_CODE.sub("", FENCES.sub("", text))
        for pattern in (MD_LINK, HTML_LINK):
            for match in pattern.finditer(text):
                url = urlsplit(html.unescape(match["url"]))
                if url.scheme or url.netloc:
                    online += 1
                    continue
                target = (root / relative).parent / unquote(url.path) if url.path else root / relative
                target = target.resolve()
                if not target.is_relative_to(root.resolve()) or not target.is_file():
                    raise ValueError(f"Broken/escaping local path in {relative}: {match['url']}")
                key = str(target.relative_to(root.resolve()))
                if url.fragment and key in documents and unquote(url.fragment) not in anchors(documents[key]):
                    raise ValueError(f"Broken local fragment in {relative}: {match['url']}")
                checked += 1
    return {"local_links_checked": checked, "online_links": online}


def verify_bundle(output, manifest):
    record = json.loads((output / "snapshot.json").read_text())
    if record["commit"] != manifest["commit"]:
        raise ValueError("Snapshot commit differs from pinned manifest")
    inventory = record["files"]
    source_entries = {"source/" + e["path"]: e for e in manifest["files"]}
    for name, entry in source_entries.items():
        if inventory.get(name) != {"bytes": entry["bytes"], "sha256": entry["sha256"]}:
            raise ValueError(f"Snapshot metadata differs from pinned source: {name}")
    actual = {str(p.relative_to(output)) for p in output.rglob("*") if p.is_file()}
    if actual != set(inventory) | {"snapshot.json"}:
        raise ValueError("Snapshot has missing or extra files")
    for name, entry in inventory.items():
        safe_path(name)
        path = output / name
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise ValueError(f"Snapshot contains a symlink: {name}")
        data = path.read_bytes()
        if len(data) != entry["bytes"] or sha256(data) != entry["sha256"]:
            raise ValueError(f"Snapshot content changed: {name}")
    docs = {name: (output / name).read_text() for name in inventory
            if name == "index.md" or name.startswith("readable/")}
    return {"commit": manifest["commit"], "files": len(actual),
            "bytes": sum((output / name).stat().st_size for name in actual),
            **check_links(output, docs)}


def write_bundle(output, files, manifest):
    rendered = {c["path"]: render_markdown(c, files[c["path"]], manifest) for c in manifest["chapters"]}
    # A few Jekyll anchors differ from Markdown anchors. Keep those as explicit
    # upstream links rather than manufacturing a broken offline fragment.
    for name, text in rendered.copy().items():
        def fix_fragment(url):
            parts = urlsplit(url)
            target = parts.path or name
            if not parts.scheme and not parts.netloc and parts.fragment and target in rendered:
                if unquote(parts.fragment) not in anchors(rendered[target]):
                    page = "" if target == "index.md" else Path(target).stem + "/"
                    return SITE + page + "#" + parts.fragment
            return url
        rendered[name] = rewrite_links(text, fix_fragment)
    index = ("# JAX Scaling Book: local source reading bundle\n\n"
             f"Upstream commit: `{manifest['commit']}`; commit date: {manifest['commit_date']}; "
             f"review date: {manifest['review_date']}.\n\n"
             "[Original authors/README](source/README.md) · [Upstream license](source/LICENSE) · "
             f"[Official website]({SITE})\n\n"
             "For local study. Full third-party image redistribution clearance is unresolved; "
             "this download does not grant it. Preserve all credits and notices.\n\n"
             "Read these Markdown files in a Markdown reader. TeX is preserved as source. "
             "Answer blocks are expanded. The three interactive plots, citation resolution, "
             "external reading, and incompatible Jekyll fragment links use the live website. "
             "No upstream scripts, fonts, styles, build tools, or executable code are installed or run.\n\n")
    for number, chapter in enumerate(manifest["chapters"]):
        name = chapter["path"]
        index += f"- {number}: [{chapter['title']}](readable/{name}) ([exact source](source/{name}))\n"
    content = {"source/" + name: data for name, data in files.items()}
    content.update({"readable/" + name: text.encode() for name, text in rendered.items()})
    content["index.md"] = index.encode()
    # mkdir is exclusive: existing user files are never overwritten.
    output.mkdir()
    for name, data in content.items():
        dest = output / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("xb") as stream:
            stream.write(data)
    inventory = {name: {"bytes": len(data), "sha256": sha256(data)} for name, data in sorted(content.items())}
    (output / "snapshot.json").write_text(json.dumps({"commit": manifest["commit"], "files": inventory}, indent=2) + "\n")
    return verify_bundle(output, manifest)


def download(archive, commit):
    url = f"https://api.github.com/repos/jax-ml/scaling-book/tarball/{commit}"
    request = Request(url, headers={"User-Agent": "OpenTPU-local-book-reader", "Accept": "application/vnd.github+json"})
    with urlopen(request, timeout=60) as response, archive.open("xb") as stream:
        total = 0
        while data := response.read(1024 * 1024):
            total += len(data)
            if total > MAX_ARCHIVE:
                raise ValueError("Download exceeds size limit")
            stream.write(data)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="new directory outside Git checkouts")
    parser.add_argument("--archive", type=Path, help="use an existing official tar.gz instead of downloading")
    parser.add_argument("--verify", action="store_true", help="check an existing bundle without network access")
    args = parser.parse_args()
    try:
        manifest = json.loads(MANIFEST.read_text())
        if not re.fullmatch(r"[0-9a-f]{40}", manifest["commit"]):
            raise ValueError("Manifest needs a full commit SHA")
        output = args.output.expanduser().resolve()
        outside_git(output)
        if args.verify:
            if args.archive:
                raise ValueError("--archive and --verify cannot be combined")
            result = verify_bundle(output, manifest)
        else:
            if output.exists() or not output.parent.is_dir():
                raise ValueError("Output must not exist, and its parent directory must exist")
            with tempfile.TemporaryDirectory(prefix="opentpu-scaling-book-") as temp:
                archive = args.archive.expanduser().resolve() if args.archive else Path(temp) / "book.tar.gz"
                if not args.archive:
                    download(archive, manifest["commit"])
                files = collect_archive(archive, manifest)
                result = write_bundle(output, files, manifest)
        print(json.dumps({"output": str(output), **result}, indent=2))
    except (OSError, ValueError, KeyError, tarfile.TarError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
