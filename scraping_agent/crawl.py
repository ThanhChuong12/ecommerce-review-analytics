"""
crawl.py — File cào dữ liệu tất-cả-trong-một cho Tiki, Lazada, TGDD.

CÁCH DÙNG:
  1. Điền danh sách URL vào URLS_TO_CRAWL bên dưới
  2. (Tùy chọn) Chỉnh MAX_REVIEWS, OUTPUT_DIR
  3. Chạy: python crawl.py

Kết quả: mỗi URL được lưu vào 1 file CSV riêng trong thư mục OUTPUT_DIR.
File tổng hợp tất cả URL cũng được tạo: OUTPUT_DIR/all_reviews_YYYYMMDD.csv

Hỗ trợ: tiki.vn | thegioididong.com | lazada.vn
"""

# ============================================================
#  ĐẶT DANH SÁCH URL CẦN CÀO VÀO ĐÂY
# ============================================================
URLS_TO_CRAWL: list[str] = [
    # --- TIKI ---
    #"https://tiki.vn/dien-thoai-samsung-galaxy-a36-5g-8gb-256gb-p277596856.html",
    #"https://tiki.vn/dien-thoai-xiaomi-redmi-15-8gb-128gb-hang-chinh-hang-p278796276.html"
    #"https://tiki.vn/son-duong-moi-hieu-chinh-ung-hong-tu-nhien-lipice-sheer-color-p33597680.html",
    #"https://tiki.vn/dau-goi-selsun-chong-gau-sach-gau-het-ngua-da-dau-selsun-anti-dandruff-shampoo-50ml-p20541866.html",
    "https://tiki.vn/thung-48-hop-sua-nestle-milo-nuoc-180ml-hop-p10240037.html",

   
    # --- TGDD ---
    #"https://www.thegioididong.com/dtdd/samsung-galaxy-a06-5g-6gb-128gb",
    #"https://www.thegioididong.com/dtdd/iphone-16-pro",
    # "https://www.thegioididong.com/dtdd/iphone-15-pro-max-1tb"
    #"https://www.thegioididong.com/dtdd/iphone-15-pro-max-1tb"

    # --- LAZADA ---
    # "https://www.lazada.vn/products/pdp-i150498381-s158167954.html"
    #"https://www.lazada.vn/products/pdp-i246452966-s316699339.html",
    #"https://www.lazada.vn/products/pdp-i150498381-s158167954.html"
    #"https://www.lazada.vn/products/pdp-i2756708-s3347924.html"
    #"https://www.lazada.vn/products/trung-nguyen-legend-ca-phe-rang-xay-sang-tao-1-bich-340gr-i353468040-s578424935.html"
    #"https://www.lazada.vn/products/pdp-i249064037-s327413856.html"
    #"https://www.lazada.vn/products/pdp-i1538401873-s6471204336.html"
    # "https://www.lazada.vn/products/pdp-i2763102767-s13733926756.html"
    # "https://www.lazada.vn/products/pdp-i1597967647-s6853265058.html"
    #"https://www.lazada.vn/products/pdp-i1465875113-s6080623320.html",
    "https://tiki.vn/thung-sua-dau-nanh-fami-nguyen-chat-200ml-x-36-hop-p12629696.html",
    "https://www.lazada.vn/products/pdp-i1465875113-s6080623320.html",
    "https://www.lazada.vn/products/pdp-i2195318921-s10444957817.html",
    "https://www.lazada.vn/products/pdp-i386626369-s6969098092.html",
    "https://www.lazada.vn/products/pdp-i2403373830-s11789805280.html",
    "https://www.lazada.vn/products/pdp-i1201294236-s4480446952.html",
    "https://www.lazada.vn/products/pdp-i1441626229-s13748512620.html",
    "https://www.lazada.vn/products/pdp-i2322467271-s14759203112.html"

]

# ============================================================
#  CẤU HÌNH
# ============================================================
MAX_REVIEWS: int     = 5_000   # Số review tối đa mỗi URL
OUTPUT_DIR: str      = "data"   # Thư mục lưu CSV (tương đối với file này)
LAZADA_HEADLESS: bool = False    # False = hiện browser để giải captcha tay nếu cần
# ============================================================

import asyncio
import os
import sys
import csv
from datetime import datetime
from pathlib import Path

# Fix encoding Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Đảm bảo import được các module trong scraping_agent/
_THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR))

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=False))

from scraper.dispatcher import scrape as _dispatch


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------

def _site_label(url: str) -> str:
    """Trả về tên site ngắn gọn để đặt tên file."""
    url = url.lower()
    if "tiki.vn" in url:
        return "tiki"
    if "thegioididong.com" in url:
        return "tgdd"
    if "lazada.vn" in url:
        return "lazada"
    return "unknown"


def _safe_filename(url: str) -> str:
    """Tạo tên file CSV an toàn từ URL."""
    import re
    label = _site_label(url)
    # Lấy slug/path cuối URL, giữ lại ký tự an toàn
    slug = url.split("//", 1)[-1]          # bỏ https://
    slug = re.sub(r"[^a-zA-Z0-9_\-]", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    slug = slug[:60]                        # giới hạn độ dài
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{label}_{slug}_{ts}.csv"


def _row_key(row: dict) -> str:
    """Tạo dedup key từ nội dung review (text + rating + date + url)."""
    import hashlib
    raw = "|".join([
        str(row.get("text", ""))[:200],
        str(row.get("rating", "")),
        str(row.get("date", "")),
        str(row.get("product_url", "")),
    ])
    return hashlib.md5(raw.encode("utf-8", errors="replace")).hexdigest()


def _load_existing_keys(path: Path) -> set[str]:
    """Đọc file all_reviews.csv hiện có và trả về set các dedup key."""
    seen: set[str] = set()
    if not path.exists():
        return seen
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                seen.add(_row_key(row))
    except Exception:
        pass
    return seen


def _append_to_master(csv_files: list[Path], out_path: Path) -> tuple[int, int]:
    """
    Append các review từ csv_files vào out_path (file tổng hợp cố định).
    Tự động bỏ qua các review đã tồn tại trong out_path (dedup).
    Trả về (số mới thêm, số bị bỏ qua do trùng).
    """
    # Đọc các key đã tồn tại trong file master
    existing_keys = _load_existing_keys(out_path)
    old_count = len(existing_keys)

    file_exists = out_path.exists()
    added = 0
    skipped = 0
    fieldnames = None

    with open(out_path, "a", newline="", encoding="utf-8-sig") as fout:
        writer = None
        for fp in csv_files:
            if not fp.exists():
                continue
            with open(fp, newline="", encoding="utf-8-sig") as fin:
                reader = csv.DictReader(fin)
                if fieldnames is None:
                    fieldnames = reader.fieldnames
                if writer is None:
                    writer = csv.DictWriter(fout, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
                    # Ghi header chỉ khi file hoàn toàn mới
                    if not file_exists:
                        writer.writeheader()
                for row in reader:
                    key = _row_key(row)
                    if key in existing_keys:
                        skipped += 1
                    else:
                        existing_keys.add(key)
                        writer.writerow(row)
                        added += 1

    return added, skipped


def _print_banner() -> None:
    bar = "=" * 65
    print(f"\n{bar}")
    print("  E-commerce Review Crawler — Tiki / TGDD / Lazada")
    print(bar)
    print(f"  URLs      : {len(URLS_TO_CRAWL)}")
    print(f"  Max/URL   : {MAX_REVIEWS:,} reviews")
    print(f"  Output    : {Path(OUTPUT_DIR).resolve()}")
    print(f"  Lazada    : headless={LAZADA_HEADLESS}")
    print(bar)
    print("  Nhấn Ctrl+C bất kỳ lúc nào để dừng — dữ liệu đã lưu sẽ giữ nguyên.\n")


# ----------------------------------------------------------------
# Core async runner
# ----------------------------------------------------------------

async def _crawl_one(
    url: str,
    output_path: Path,
    idx: int,
    total: int,
) -> int:
    """Cào một URL, ghi ra output_path. Trả về số review đã lưu."""
    bar = "-" * 65
    print(f"\n{bar}")
    print(f"  [{idx}/{total}] {_site_label(url).upper()} → {url[:80]}")
    print(f"  Đang lưu vào: {output_path.name}")
    print(bar)

    # Truyền headless cho Lazada qua dispatcher; dispatcher đọc 'headless' argument
    total_saved = await _dispatch(
        url=url,
        output_path=str(output_path),
        fmt="csv",
        max_reviews=MAX_REVIEWS,
        llm_provider="auto",    # chỉ dùng Approach 1 — không cần LLM
        headless=LAZADA_HEADLESS,
    )
    return total_saved


async def main() -> None:
    if not URLS_TO_CRAWL:
        print("\n[!] URLS_TO_CRAWL đang trống!")
        print("    Mở file crawl.py và điền ít nhất 1 URL vào danh sách rồi chạy lại.\n")
        sys.exit(1)

    _print_banner()

    out_dir = (_THIS_DIR / OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []   # {"url", "file", "count", "status"}
    csv_files: list[Path] = []

    for idx, url in enumerate(URLS_TO_CRAWL, start=1):
        out_file = out_dir / _safe_filename(url)
        try:
            count = await _crawl_one(url, out_file, idx, len(URLS_TO_CRAWL))
            results.append({"url": url, "file": out_file.name, "count": count, "status": "OK"})
            csv_files.append(out_file)
        except KeyboardInterrupt:
            print("\n[!] Dừng theo yêu cầu người dùng.")
            results.append({"url": url, "file": out_file.name, "count": 0, "status": "STOPPED"})
            break
        except Exception as exc:
            print(f"\n[!] Lỗi khi cào {url}: {exc}")
            results.append({"url": url, "file": out_file.name, "count": 0, "status": f"ERROR: {exc}"})

    # ── Append vào file tổng hợp cố định ────────────────────────────────
    MASTER_FILE = "all_reviews.csv"
    merged_path = out_dir / MASTER_FILE
    added, skipped = _append_to_master(csv_files, merged_path)
    print(f"\n  [all_reviews.csv] +{added:,} mới, bỏ qua {skipped:,} trùng → {merged_path}")

    # ── In bảng tổng kết ──────────────────────────────────────────────
    bar = "=" * 65
    print(f"\n{bar}")
    print("  KẾT QUẢ TỔNG KẾT")
    print(bar)
    print(f"  {'URL':<55} {'Reviews':>8}  Status")
    print(f"  {'-'*54} {'-'*8}  {'-'*10}")
    for r in results:
        url_short = r["url"][:54]
        print(f"  {url_short:<55} {r['count']:>8,}  {r['status']}")
    total_reviews = sum(r["count"] for r in results)
    print(f"  {'':55} {'─'*8}")
    print(f"  {'TỔNG':<55} {total_reviews:>8,}")
    print(bar)
    print(f"  Tất cả file CSV đã lưu vào : {out_dir.resolve()}")
    print(f"  File tổng hợp (cộng dồn)   : {MASTER_FILE}")
    print(f"{bar}\n")


if __name__ == "__main__":
    asyncio.run(main())
