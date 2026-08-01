"""Prepare reviewed OMT prose without images, scores, embeds or whole-book ingestion.

The committed manifest contains URLs and reviewed paragraph hashes, never book
text. Setup is explicit and network-bound; product requests only read its local
JSON. A changed/missing paragraph or license fails setup before replacing data.
"""
from __future__ import annotations

import hashlib
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import urlopen

MANIFEST = Path(__file__).with_name("omt_manifest.json")
DEFAULT_CORPUS = Path(__file__).resolve().parents[1] / "data/openmusictheory/reviewed.json"
LICENSE_URL = "https://creativecommons.org/licenses/by-sa/4.0/"


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf8")).hexdigest()


class ProseParser(HTMLParser):
    """Plain paragraphs inside an article, with no quoted/embedded third-party material."""

    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
    EXCLUDE = {"blockquote", "table", "figure", "iframe", "script", "style", "pre", "audio", "video"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.blocks: list[dict[str, str]] = []
        self.section = ""
        self.capture: str | None = None
        self.parts: list[str] = []
        self.rejected = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if self.capture and (tag in self.EXCLUDE or tag in {"img", "embed"}):
            self.rejected = True
        if tag not in self.VOID:
            self.stack.append(tag)
        if "article" not in self.stack or any(t in self.EXCLUDE for t in self.stack):
            return
        if tag in {"p", "h2", "h3"}:
            self.capture, self.parts, self.rejected = tag, [], False
        if self.capture and tag == "a":
            href = attrs.get("href", "")
            if href.startswith(("http:", "https:")) and not href.startswith("https://openmusictheory.github.io/"):
                self.rejected = True
        if tag == "br" and self.capture:
            self.parts.append(" ")

    def handle_data(self, data):
        if self.capture:
            self.parts.append(data)

    def handle_endtag(self, tag):
        if tag == self.capture:
            text = re.sub(r"\s+", " ", "".join(self.parts)).strip()
            if not self.rejected and text:
                if tag in {"h2", "h3"}:
                    self.section = text
                elif len(text.split()) >= 12:
                    self.blocks.append({"section": self.section, "text": text, "sha256": digest(text)})
            self.capture = None
        if tag in self.stack:
            self.stack = self.stack[:len(self.stack) - 1 - self.stack[::-1].index(tag)]


def parse_page(html: str) -> list[dict[str, str]]:
    if "creativecommons.org/licenses/by-sa/4.0/" not in html:
        raise ValueError("OMT page has no expected CC BY-SA 4.0 license link")
    parser = ProseParser()
    parser.feed(html)
    return parser.blocks


def download(url: str) -> str:
    if not re.fullmatch(r"https://openmusictheory\.github\.io/[A-Za-z0-9-]+\.html", url):
        raise ValueError("Only reviewed OMT publisher URLs can be downloaded")
    with urlopen(url, timeout=20) as response:
        if response.url != url:
            raise ValueError("OMT source redirected; review the new source before ingestion")
        data = response.read(2_000_001)
    if len(data) > 2_000_000:
        raise ValueError("OMT page exceeds the setup download limit")
    return data.decode("utf8")


def prepare(manifest: dict, fetch=download) -> dict:
    passages = []
    for page in manifest["pages"]:
        blocks = {block["sha256"]: block for block in parse_page(fetch(page["url"]))}
        for expected in page["paragraphs"]:
            block = blocks.get(expected)
            if block is None:
                raise ValueError(f"Reviewed OMT paragraph changed or missing: {page['url']} ({expected})")
            passages.append({
                **block, "id": expected, "source": "omt", "title": page["title"],
                "url": page["url"], "authors": manifest["authors"],
                "editors": manifest["editors"],
                "license": "CC BY-SA 4.0", "license_url": LICENSE_URL,
                "copyright": manifest["copyright"],
                "changes": "Selected prose; HTML formatting normalized; media excluded.",
            })
    if not passages:
        raise ValueError("No reviewed OMT paragraphs selected")
    return {"version": 1, "source": "omt", "edition": manifest["edition"], "passages": passages}


def setup() -> int:
    manifest = json.loads(MANIFEST.read_text(encoding="utf8"))
    corpus = prepare(manifest)
    DEFAULT_CORPUS.parent.mkdir(parents=True, exist_ok=True)
    # Same-directory atomic replacement, no temporary directory or partial corpus.
    staging = DEFAULT_CORPUS.with_suffix(".pending")
    try:
        staging.write_text(json.dumps(corpus, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
        staging.replace(DEFAULT_CORPUS)
    finally:
        staging.unlink(missing_ok=True)
    print(f"OMT ready: {len(corpus['passages'])} reviewed passages; Berklee untouched.")
    return len(corpus["passages"])
