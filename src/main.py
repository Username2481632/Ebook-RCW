import base64
import datetime
import multiprocessing
import os
import shutil
import signal
import threading
import typing
import urllib.parse

import ebooklib.epub
import lxml.html
import requests
import scrapy.crawler
import scrapy.utils.defer
import tqdm
import twisted.internet

import constants


def create_title(url: str) -> str:
    return f"{url.split('=')[-1]}{' (dispo)' if url.split('/')[-1].startswith('dispo') else ''}"


def create_filename(url: str) -> str:
    return f"{encode_id(url)}.xhtml"


def encode_id(url: str) -> str:
    """Encode a URL to a safe EPUB ID or filename."""
    # URL-safe Base64 (no padding, so it's filesystem-friendly)
    return base64.urlsafe_b64encode(url.encode("utf-8")).decode("ascii").rstrip("=")


def decode_id(idref: str) -> str:
    """Decode an IDref to the original URL."""
    # Restore padding for base64
    padded = idref + "=" * (-len(idref) % 4)
    return base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
 
_ENCODED_HTTP_PREFIX = encode_id("htt")
def is_encoded_url(value: str | ebooklib.epub.EpubItem) -> bool:
    if isinstance(value, str):
        return value.startswith(_ENCODED_HTTP_PREFIX)
    if isinstance(value, ebooklib.epub.EpubItem):
        return not isinstance(value, ebooklib.epub.EpubNcx) and not isinstance(value, ebooklib.epub.EpubNav) and not isinstance(value, ebooklib.epub.EpubCoverHtml)
    raise TypeError("Expected str or EpubItem.")


def get_version() -> datetime.date:
    response: requests.Response = requests.get(constants.VERSION_URL)
    response.raise_for_status()
    tree: lxml.html.HtmlElement = lxml.html.fromstring(response.content)

    date_text: list[str] = tree.xpath(
        "/html/body/div[2]/div[2]/div/div[2]/div[1]/b/text()"
    )
    if not date_text:
        raise ValueError("Could not find last updated date.")

    raw_text: str = date_text[0]
    if not raw_text.startswith("Last Update: "):
        raise ValueError(f"Unexpected date text format: {raw_text}")

    date_part: str = raw_text.replace("Last Update: ", "")
    parsed_date: datetime.date = datetime.datetime.strptime(
        date_part, "%B %d, %Y"
    ).date()

    return parsed_date


def create_link(page_id: str) -> ebooklib.epub.Link:
    return ebooklib.epub.Link(
        create_filename(decode_id(page_id)), create_title(decode_id(page_id)), page_id
    )


TOCSectionType = (
    typing.Tuple[typing.Union[ebooklib.epub.Link, "TOCSectionType"], ...]
    | ebooklib.epub.Link
)
TreeType = list[TOCSectionType]


def create_toc(spine: list[tuple[str, dict[str, str]]]) -> TreeType:
    index = 0
    spine.sort(
        key=lambda x: decode_id(x[0]) if x[0].startswith(encode_id("htt")) else ""
    )

    def worker(parent_title: str = "") -> TreeType:
        nonlocal index
        tree: TreeType = []
        while index != len(spine):
            if not is_encoded_url(spine[index][0]):
                index += 1
                continue

            assert index < len(spine), (
                f"Index {index} out of bounds for {len(spine)} documents."
            )

            current_url: str = decode_id(spine[index][0])
            if current_url.startswith(parent_title):
                index += 1
                if (
                    index != len(spine)
                    and is_encoded_url(spine[index][0])
                    and decode_id(spine[index][0]).startswith(current_url)
                ):
                    tree.append(
                        (
                            create_link(spine[index - 1][0]),
                            tuple(worker(current_url)),
                        )
                    )
                else:
                    tree.append(create_link(spine[index - 1][0]))
            else:
                break
        return tree

    return worker()


def include_url(url: str, parent_url: str | None = None) -> bool:
    return (
        url.split("://")[1]
        .lower()
        .startswith("app.leg.wa.gov/rcw/default.aspx?cite=46")
    )


def extract_requests(
    tree: lxml.html.HtmlElement, base_url: str | None = None
) -> list[scrapy.Request]:
    """If base_url is provided, assumes that href replacement is needed."""
    new_requests: list[scrapy.Request] = []
    for element in tree.iter("a"):
        href: str | None = element.get("href")
        assert href
        if base_url:
            href = urllib.parse.urljoin(base_url, href)
        elif href.endswith(".xhtml"):
            href = decode_id(href.removesuffix(".xhtml"))

        if include_url(href, base_url):
            if href.startswith("http://"):
                href = href.replace("http://", "https://", 1)
            href = href.lower()
            if base_url:
                element.set("href", create_filename(href))
            new_requests.append(scrapy.Request(href))
        elif base_url:
            element.set("href", href)

    return new_requests


class EpubSpider(scrapy.Spider):
    name = "epub_creator"

    async def start(self):
        self.book: ebooklib.epub.EpubBook
        if os.path.exists(constants.OUTPUT):
            self.book = ebooklib.epub.read_epub(constants.OUTPUT)
            version: datetime.date = self.book.get_metadata("DC", "date")[0][0]
            assert version
            if version != get_version().isoformat():
                self.book = ebooklib.epub.EpubBook()
        else:
            self.book = ebooklib.epub.EpubBook()
        self.book.set_identifier("af101ccd-dd08-4c89-aec8-94fdd041bd95")
        self.book.set_title("Revised Code of Washington - Title 46")
        self.book.set_language("en")
        self.book.add_author("Washington State Legislature")
        self.book.add_metadata("DC", "date", get_version().isoformat())
        with open(constants.COVER_IMAGE, "rb") as cover_file:
            self.book.set_cover("cover.jpg", cover_file.read())

        initial_requests: list[scrapy.Request] = (
            [scrapy.Request(constants.TARGET_URL)] if not self.book.spine else []
        )

        assert self.crawler.engine
        assert self.crawler.engine._slot

        book_items: dict[scrapy.Request, str] = {}
        for item in self.book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            if is_encoded_url(item):
                idref = item.get_id()
                print(idref)

                req: scrapy.Request = scrapy.Request(decode_id(idref))
                fp = self.crawler.engine._slot.scheduler.df.request_fingerprint(req)
                assert req not in book_items, f"Book contains duplicate item: {req.url}"
                book_items[req] = fp

        queue_copy: list[scrapy.Request] = []
        pending: set[str] = set()

        assert hasattr(self.crawler.engine._slot.scheduler, "dqs")
        old_handler: signal.Handlers = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self.forced_cleanup)
        if self.crawler.engine._slot.scheduler.dqs:
            while True:
                r = self.crawler.engine._slot.scheduler.dqs.pop()
                if r is None:
                    break
                queue_copy.append(r)
                pending.add(
                    self.crawler.engine._slot.scheduler.df.request_fingerprint(r)
                )

        scrapy_fingerprints: set[str] = (
            self.crawler.engine._slot.scheduler.df.fingerprints - pending
        )
        book_fingerprints: set[str] = set(book_items.values())
        if book_fingerprints != scrapy_fingerprints:
            extra_in_book = book_fingerprints - scrapy_fingerprints
            extra_in_scheduler = scrapy_fingerprints - book_fingerprints
            print("Mismatched progress detected—clearing request queue.")
            print(
                f"Total dupefilter fingerprints: {len(self.crawler.engine._slot.scheduler.df.fingerprints)}"
            )
            for name, extras in (
                ("book", extra_in_book),
                ("scheduler", extra_in_scheduler),
            ):
                print(
                    f"Extra in {name}: {list(extras) if len(extras) < 2 else f'[{"; ".join(list(extras)[:2] + [f"... ({len(extras)} total)"])})]'}"
                )
            with open(
                os.path.join(self.crawler.settings.get("JOBDIR"), "requests.seen"),
                "w",
                encoding="utf-8",
            ) as f:
                f.writelines(f"{fp}\n" for fp in book_fingerprints)
            self.crawler.engine._slot.scheduler.df.fingerprints = book_fingerprints

            # Iterate through stored_in_book directly
            for req in book_items:
                initial_requests.extend(
                    extract_requests(
                        lxml.html.fromstring(
                            self.book.get_item_with_id(
                                encode_id(req.url)
                            ).get_body_content()
                        )
                    )
                )
        else:
            # put them back into the disk queue
            for r in queue_copy:
                self.crawler.engine._slot.scheduler.dqs.push(r)

        self.book_lock = threading.Lock()
        self.seen_lock = threading.Lock()
        self.pbar = tqdm.tqdm(
            desc="Crawling",
            total=len(book_items) + len(self.crawler.engine._slot.scheduler),
            unit="pg",
            initial=len(book_items),
        )

        signal.signal(signal.SIGINT, old_handler)
        for req in initial_requests:
            yield req

    def process_html(self, response: scrapy.http.Response) -> list[scrapy.Request]:
        if response.status in constants.REDIRECT_RESPONSE_CODES:
            redirected_url: str = response.headers.get("Location", "").decode("utf-8")
            assert redirected_url, f"Redirected URL is empty on url {response.url}"
            redirected_url = urllib.parse.urljoin(response.url, redirected_url)
            assert redirected_url.startswith("http")
            alias_page: ebooklib.epub.EpubHtml = ebooklib.epub.EpubHtml(
                title=create_title(response.url),
                file_name=create_filename(response.url),
                uid=encode_id(response.url),
            )
            alias_page.content = f"""<!DOCTYPE html>
<html>
<head>
    <meta http-equiv="refresh" content="0; url='{create_filename(redirected_url)}'" />
</head>
<body>
  <p>If you are not redirected automatically, <a href="{create_filename(redirected_url)}">click here</a>.</p>
</body>
</html>
"""
            self.book.add_item(alias_page)
            self.pbar.update(1)

            return [scrapy.Request(redirected_url)]

        tree: lxml.html.HtmlElement = lxml.html.fromstring(response.text)
        content: lxml.html.HtmlElement = tree.xpath('//*[@id="contentWrapper"]')
        if not content:
            content = tree.xpath('//*[@id="divContent"]')[0]
            content.remove(content[0])
        else:
            content = content[0]

        for element in content.xpath('.//*[contains(@class, "ui-btn")]'):
            element.getparent().remove(element)

        new_requests: list[scrapy.Request] = extract_requests(content, response.url)
        assert all(
            element.get("href").endswith(".xhtml")
            or element.get("href").startswith("http")
            for element in content.xpath(".//a")
        )

        page: ebooklib.epub.EpubHtml = ebooklib.epub.EpubHtml(
            title=create_title(response.url),
            file_name=create_filename(response.url),
            uid=encode_id(response.url),
        )
        page.content = lxml.html.tostring(content, encoding="unicode")

        with self.book_lock:
            self.book.add_item(page)
            self.book.spine.append((page.id, "yes"))
        if not self.crawler.crawling:
            self.pbar.set_description("Saving state")
        assert self.crawler.engine
        assert self.crawler.engine._slot
        self.pbar.total = len(self.crawler.engine._slot.scheduler.df.fingerprints)
        self.pbar.update(1)

        return new_requests

    def parse(self, response: scrapy.http.Response) -> None:
        return twisted.internet.threads.deferToThread(self.process_html, response)

    def closed(self, _reason: str):
        if not self.book.spine:
            print("Book empty. No EPUB will be created.")
            return

        self.book.toc = create_toc(self.book.spine)
        self.book.items = list(
            filter(
                lambda x: not isinstance(x, ebooklib.epub.EpubNcx)
                and not isinstance(x, ebooklib.epub.EpubNav),
                self.book.items,
            )
        )
        self.book.add_item(ebooklib.epub.EpubNcx())  # old-style EPUB 2 navigation (ncx)
        self.book.add_item(
            ebooklib.epub.EpubNav()
        )  # modern EPUB 3 navigation (nav.xhtml)

        if not self.book.get_metadata("DC", "date")[0][0] == get_version().isoformat():
            print(
                f"WARNING: **RCW version changed from {self.book.get_metadata('DC', 'date')[0][0]} to {get_version().isoformat()} mid-crawl. This may result in an invalid ebook.**"
            )

        # Assert the book has at least one item
        assert self.book.items
        assert self.book.spine

        if ("nav", "yes") not in self.book.spine:
            self.book.spine.insert(0, ("nav", "yes"))

        self.writer_process = multiprocessing.Process(
            target=ebooklib.epub.write_epub,
            args=(constants.OUTPUT, self.book),
            daemon=True,
        )
        signal.signal(signal.SIGINT, self.forced_cleanup)
        self.pbar.set_description("Writing output")
        self.writer_process.start()
        self.writer_process.join()

    def forced_cleanup(self, _sig: int, _frame: int):
        if hasattr(self, "writer_process") and self.writer_process.is_alive():
            self.writer_process.kill()
        if os.path.exists(constants.OUTPUT):
            os.remove(constants.OUTPUT)
        if os.path.exists(self.crawler.settings.get("JOBDIR")):
            shutil.rmtree(self.crawler.settings.get("JOBDIR"))


if __name__ == "__main__":
    process = scrapy.crawler.CrawlerProcess(
        settings={
            "JOBDIR": os.path.abspath(constants.OUTPUT) + ".jobdir",
            "LOG_LEVEL": "WARNING",
            "HTTPCACHE_ENABLED": True,
            "CONCURRENT_REQUESTS_PER_DOMAIN": constants.SIMULTANEOUS_REQUESTS,
            "REACTOR_THREADPOOL_MAXSIZE": constants.NUM_WORKERS,
            "REDIRECT_ENABLED": False,
            "HTTPERROR_ALLOWED_CODES": constants.REDIRECT_RESPONSE_CODES,
            "CLOSESPIDER_ERRORCOUNT": 1,
        }
    )
    process.crawl(EpubSpider)
    process.start()
