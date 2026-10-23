"""Downloader integrity and filesystem boundary tests; no network or hardware."""

import io
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

import download_scaling_book as book


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.data = b"The upstream license fixture\n"
        self.manifest = {"files": [{"path": "LICENSE", "bytes": len(self.data),
                                    "sha256": book.sha256(self.data)}]}

    def archive(self, entries):
        path = self.root / "book.tar.gz"
        with tarfile.open(path, "w:gz") as tar:
            for name, data, kind in entries:
                item = tarfile.TarInfo(name)
                item.type = kind
                if kind == tarfile.REGTYPE:
                    item.size = len(data)
                    tar.addfile(item, io.BytesIO(data))
                else:
                    item.linkname = "../../outside"
                    tar.addfile(item)
        return path

    def test_allowlist_excludes_scripts(self):
        archive = self.archive([("upstream/LICENSE", self.data, tarfile.REGTYPE),
                                ("upstream/install.sh", b"do not execute", tarfile.REGTYPE)])
        self.assertEqual(book.collect_archive(archive, self.manifest), {"LICENSE": self.data})

    def test_changed_content_rejected(self):
        archive = self.archive([("upstream/LICENSE", b"x" * len(self.data), tarfile.REGTYPE)])
        with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
            book.collect_archive(archive, self.manifest)

    def test_missing_file_rejected(self):
        archive = self.archive([("upstream/other.md", b"unselected", tarfile.REGTYPE)])
        with self.assertRaisesRegex(ValueError, "Missing selected"):
            book.collect_archive(archive, self.manifest)

    def test_traversal_absolute_symlink_hardlink_rejected(self):
        for name, kind in [("upstream/../escape", tarfile.REGTYPE),
                           ("/absolute", tarfile.REGTYPE),
                           ("upstream/link", tarfile.SYMTYPE),
                           ("upstream/hardlink", tarfile.LNKTYPE)]:
            with self.subTest(name=name):
                archive = self.archive([("upstream/LICENSE", self.data, tarfile.REGTYPE),
                                        (name, b"bad", kind)])
                with self.assertRaises(ValueError):
                    book.collect_archive(archive, self.manifest)

    def test_git_worktree_boundary(self):
        (self.root / ".git").write_text("gitdir: /irrelevant")
        with self.assertRaisesRegex(ValueError, "outside every Git"):
            book.outside_git(self.root / "download")

    def test_duplicate_paths_rejected(self):
        archive = self.archive([("upstream/LICENSE", self.data, tarfile.REGTYPE)] * 2)
        with self.assertRaisesRegex(ValueError, "duplicate archive path"):
            book.collect_archive(archive, self.manifest)

    def test_file_size_limit(self):
        archive = self.archive([("upstream/LICENSE", self.data, tarfile.REGTYPE)])
        with patch.object(book, "MAX_FILE", 1):
            with self.assertRaisesRegex(ValueError, "size limit"):
                book.collect_archive(archive, self.manifest)

    def test_hlo_braces_in_code_are_preserved(self):
        source = b"---\ntitle: HLO\n---\n{% raw %}`replica_groups={{0,16}, ...}`{% endraw %}\n"
        chapter = {"path": "profiling.md", "title": "HLO"}
        result = book.render_markdown(chapter, source,
                                      {"chapters": [chapter], "files": [], "commit": "a" * 40})
        self.assertIn("`replica_groups={{0,16}, ...}`", result)
        self.assertNotIn("{%", result)

    def test_local_links_and_fragments(self):
        docs = {"index.md": "# Index\n[Chapter](chapter.md#topic)\n", "chapter.md": "# Topic\n"}
        for name, text in docs.items():
            (self.root / name).write_text(text)
        self.assertEqual(book.check_links(self.root, docs)["local_links_checked"], 1)
        docs["index.md"] = "[Bad](chapter.md#missing)"
        with self.assertRaisesRegex(ValueError, "Broken local fragment"):
            book.check_links(self.root, docs)

    def test_bundle_preserves_source_and_detects_tampering(self):
        source = b"---\ntitle: Intro\n---\n## Topic\nRead the source.\n"
        files = {"LICENSE": self.data, "README.md": b"Original attribution\n", "index.md": source}
        manifest = {"commit": "a" * 40, "commit_date": "2026-10-19", "review_date": "2026-10-23",
                    "chapters": [{"path": "index.md", "title": "Intro"}],
                    "files": [{"path": n, "bytes": len(d), "sha256": book.sha256(d)} for n, d in files.items()]}
        output = self.root / "output"
        result = book.write_bundle(output, files, manifest)
        self.assertEqual(result["files"], 6)
        self.assertEqual((output / "source/index.md").read_bytes(), source)
        with self.assertRaises(FileExistsError):
            book.write_bundle(output, files, manifest)
        (output / "source/LICENSE").write_text("changed")
        with self.assertRaisesRegex(ValueError, "Snapshot content changed"):
            book.verify_bundle(output, manifest)


if __name__ == "__main__":
    unittest.main()
