"""
Core scraping agent — uses browser_use.Agent directly (no custom wrappers).

Improvements v3:
  * Auto-fallback LLM: if primary hits rate-limit, switch to backup automatically
  * Filter guard: agent explicitly told NOT to click star/filter tabs on Shopee/Lazada
  * Human-like scrolling: small increments with waits to avoid bot detection
  * Full-size images: strip thumbnail suffixes from Shopee/Lazada image URLs
  * Random wait after pagination to mimic human behavior
  * Deduplication: md5(text+date+rating) to prevent duplicate rows
  * max_history_items=15 to keep token cost low on long runs
"""

import hashlib
import os
import random
from datetime import datetime

from dotenv import load_dotenv
from pydantic import BaseModel, Field

from browser_use import ActionResult, Agent, BrowserProfile, BrowserSession, Controller

from scraper.exporter import ReviewExporter
from scraper.models import Review

load_dotenv()

# ---------------------------------------------------------------------------
# LLM factory — primary + automatic fallback
# ---------------------------------------------------------------------------


def get_llm(provider: str = 'auto'):
    """Return the best available chat model. Also returns a fallback if possible."""

    if provider == 'auto':
        if os.getenv('GOOGLE_API_KEY'):
            provider = 'google'
        elif os.getenv('BROWSER_USE_API_KEY'):
            provider = 'browseruse'
        elif os.getenv('OPENAI_API_KEY'):
            provider = 'openai'
        elif os.getenv('GROQ_API_KEY'):
            provider = 'groq'
        else:
            raise ValueError(
                'No API key found. Add one of these to .env:\n'
                '  GOOGLE_API_KEY=...      (recommended)\n'
                '  BROWSER_USE_API_KEY=... (fastest)\n'
                '  OPENAI_API_KEY=...\n'
                '  GROQ_API_KEY=...        (free but rate-limited)\n'
            )

    if provider == 'google':
        from browser_use import ChatGoogle
        print('LLM: Google Gemini 2.0 Flash')
        primary = ChatGoogle(model='gemini-2.0-flash')
        # Fallback: groq if available
        fallback = _try_groq_fallback()
        return primary, fallback

    elif provider == 'browseruse':
        from browser_use import ChatBrowserUse
        print('LLM: ChatBrowserUse (recommended)')
        primary = ChatBrowserUse()
        fallback = _try_groq_fallback()
        return primary, fallback

    elif provider == 'openai':
        from browser_use import ChatOpenAI
        print('LLM: OpenAI gpt-4.1-mini')
        primary = ChatOpenAI(model='gpt-4.1-mini')
        fallback = _try_groq_fallback()
        return primary, fallback

    elif provider == 'groq':
        from browser_use import ChatGroq
        print('LLM: Groq llama-4-scout')
        primary = ChatGroq(model='meta-llama/llama-4-scout-17b-16e-instruct')
        # Fallback: google if available (groq is often rate-limited)
        fallback = _try_google_fallback()
        if fallback:
            print('  Fallback LLM: Google Gemini (auto-switch on rate-limit)')
        return primary, fallback

    else:
        raise ValueError(f'Unknown provider: {provider}')


def _try_groq_fallback():
    if os.getenv('GROQ_API_KEY'):
        from browser_use import ChatGroq
        return ChatGroq(model='meta-llama/llama-4-scout-17b-16e-instruct')
    return None


def _try_google_fallback():
    if os.getenv('GOOGLE_API_KEY'):
        from browser_use import ChatGoogle
        return ChatGoogle(model='gemini-2.0-flash')
    return None


# ---------------------------------------------------------------------------
# Task prompt — v3 (filter-aware, scroll-aware, image-aware)
# ---------------------------------------------------------------------------


def build_task(url: str, max_reviews: int) -> str:
    return f"""
PRODUCT URL: {url}
TARGET: Collect up to {max_reviews} product reviews.

=== STRICT RULES ===

RULE 1 — NAVIGATE.
  Open: {url}
  Close any popup/login overlay immediately.
  Do NOT navigate away from this product page.

RULE 2 — SCROLL TO REVIEW SECTION.
  DO NOT scroll 1 page at a time repeatedly — this will loop forever.
  Instead, use JavaScript to jump directly to the review section:
    execute_javascript("
      const sel = ['[data-tracker-section=\"ratings\"]',
                    '#review', '.product-ratings', '[class*=\"review\"]',
                    '[class*=\"rating\"]', 'section.rating'];
      for (const s of sel) {{
        const el = document.querySelector(s);
        if (el) {{ el.scrollIntoView({{behavior:'smooth',block:'center'}}); break; }}
      }}
      // Fallback: scroll to 80% of page height
      if (!document.querySelector('[class*=\"review\"],[class*=\"rating\"]'))
        window.scrollTo(0, document.body.scrollHeight * 0.8);
    ")
  Wait 2 seconds for the review section to load.
  If still not visible, scroll to bottom once more then wait 2s.
  Do NOT keep scrolling in a loop — try max 2 scroll attempts then proceed.

RULE 3 — FILTER: SELECT "TẤT CẢ" (ALL).
  Click the "Tất Cả" tab inside the review section to show ALL reviews.
  DO NOT click any star filters (5★, 4★ etc.) — they hide reviews.

RULE 4 — CALL extract_and_save_reviews ONCE PER PAGE.
  This is your ONLY tool for saving reviews.
  For each review card visible on the page, pass:
    text: full customer comment (skip only if truly empty)
    rating: count FILLED YELLOW stars only (1-5)
    date: the date shown on the review
    image_urls: list of image URLs on the review (empty list if none)
  Call this tool ONCE with ALL reviews on the current page.

RULE 5 — PAGINATE.
  After extract_and_save_reviews returns:
  - If it says "STOP" or "TARGET REACHED" → you are done, call done().
  - Otherwise, find the pagination (← 1 2 3 → buttons) inside the review
    section, click the NEXT page number, wait 2 seconds, then repeat RULE 4.
  - If there is no next page button, call done().

=== START: Navigate to the URL, follow rules 1→5. ===
""".strip()


# ---------------------------------------------------------------------------
# Input model for save_reviews action
# ---------------------------------------------------------------------------


class ReviewsInput(BaseModel):
    reviews: list[dict] = Field(
        description=(
            'List of review dicts extracted from the CURRENT page. '
            'Each dict must have: text (str), rating (int 1-5), '
            'date (str), image_urls (list of str).'
        )
    )
    page_number: int = Field(default=1, description='Current page number (1-indexed)')
    has_more: bool = Field(default=True, description='True if a next page exists')


# ---------------------------------------------------------------------------
# Main scraping coroutine
# ---------------------------------------------------------------------------


async def scrape_reviews(
    url: str,
    output_path: str,
    fmt: str = 'csv',
    max_reviews: int = 3000,
    llm_provider: str = 'auto',
    headless: bool = False,
) -> int:
    """Run browser_use.Agent to scrape product reviews from any URL."""

    llm, fallback_llm = get_llm(llm_provider)
    exporter = ReviewExporter(output_path, fmt)

    # Shared state (closure)
    class _State:
        total: int = 0
        target: int = max_reviews
        seen: set = set()

    state = _State()

    # ---------------------------------------------------------------------------
    # Custom action: save_reviews
    # ---------------------------------------------------------------------------

    controller = Controller()

    @controller.action(
        'Extract reviews visible on the current page AND save them. '
        'Call this ONCE per page after all reviews are visible. '
        'Pass ALL reviews from the page in the reviews list.',
        param_model=ReviewsInput,
    )
    async def extract_and_save_reviews(params: ReviewsInput) -> ActionResult:
        new_reviews: list[Review] = []

        for raw in params.reviews:
            # Skip reviews with no text
            text = str(raw.get('text') or '').strip()
            if not text:
                continue

            try:
                review = Review(
                    text=text,
                    rating=raw.get('rating', 5),
                    date=raw.get('date', ''),
                    image_urls=_process_image_urls(raw.get('image_urls', [])),
                    product_url=url,
                    scraped_at=datetime.now().isoformat(),
                )

                # Dedup: skip if already saved
                key = hashlib.md5(
                    f'{review.text[:80]}|{review.date}|{review.rating}'.encode()
                ).hexdigest()
                if key in state.seen:
                    continue
                state.seen.add(key)
                new_reviews.append(review)

            except Exception as e:
                print(f'    Skip invalid review: {e}')

        saved = exporter.save_batch(new_reviews)
        state.total += saved
        skipped = len(params.reviews) - saved

        print(
            f'  [OK] Page {params.page_number}: +{saved} reviews'
            + (f' (skip {skipped} dupes/empty)' if skipped else '')
            + f' — Total: {state.total:,}/{state.target:,}'
        )

        if state.total >= state.target:
            return ActionResult(
                extracted_content=f'TARGET REACHED: {state.total} reviews saved. Call done() now.',
                is_done=True,
            )

        if not params.has_more:
            return ActionResult(
                extracted_content=f'No more pages. Total: {state.total} reviews.',
                is_done=True,
            )

        # Random human-like wait hint for the agent
        wait_hint = round(random.uniform(1.0, 3.0), 1)
        return ActionResult(
            extracted_content=(
                f'Saved {saved} reviews from page {params.page_number}. '
                f'Running total: {state.total}/{state.target}. '
                f'Wait {wait_hint}s, then click page {params.page_number + 1} in the review pagination.'
            )
        )

    # ---------------------------------------------------------------------------
    # Browser + Agent
    # ---------------------------------------------------------------------------

    browser_profile = BrowserProfile(
        headless=headless,
        user_data_dir='./browser_data',
    )
    browser_session = BrowserSession(browser_profile=browser_profile)

    initial_actions = [
        {'navigate': {'url': url, 'new_tab': False}},
    ]

    agent_kwargs = dict(
        task=build_task(url, max_reviews),
        llm=llm,
        tools=controller,
        browser=browser_session,
        initial_actions=initial_actions,
        max_actions_per_step=4,
        max_failures=5,
        max_history_items=15,
        use_vision=True,
        extend_system_message=(
            'CRITICAL — READ BEFORE ACTING:\n'
            '1. Your ONLY tool to save data is: extract_and_save_reviews.\n'
            '   Call it ONCE per page. Pass ALL reviews from that page.\n'
            '   Do NOT use the generic extract tool to collect reviews.\n'
            '2. NEVER navigate away from the original product URL.\n'
            '3. RATING: Count ONLY filled yellow/orange stars (ignore grey).\n'
            '   Example: 4 filled + 1 grey = rating 4.\n'
            '4. FILTER: Click "Tất Cả" (All) in review section.\n'
            '   NEVER click star-filter tabs (5★, 4★ etc.).\n'
            '5. PAGINATION: After saving, click the next numbered page button\n'
            '   (2, 3, 4...) inside the review section. Wait 2s then save again.\n'
            '6. SCROLL: 300-400px at a time with 1s pauses.\n'
            '7. If extract_and_save_reviews returns STOP or TARGET REACHED → call done().'
        ),
    )

    # Add fallback LLM if available (auto-switches on rate limit/error)
    if fallback_llm is not None:
        agent_kwargs['fallback_llm'] = fallback_llm

    agent = Agent(**agent_kwargs)

    try:
        await agent.run(max_steps=200)
        if hasattr(agent, 'is_using_fallback_llm') and agent.is_using_fallback_llm:
            print(f'  Note: Switched to fallback LLM during run.')
    except KeyboardInterrupt:
        print('\nStopped by user.')
    except Exception as e:
        print(f'\nAgent error: {e}')
    finally:
        try:
            await browser_session.stop()
        except Exception:
            pass

    return state.total


# ---------------------------------------------------------------------------
# Image URL processing — convert thumbnails to full-size
# ---------------------------------------------------------------------------


def _process_image_urls(raw_urls) -> list[str]:
    """Clean and upgrade image URLs to full size."""
    if not raw_urls:
        return []

    if isinstance(raw_urls, str):
        raw_urls = [raw_urls]

    result = []
    for url in raw_urls:
        url = str(url).strip()
        if not url:
            continue

        # Shopee thumbnail → full size (remove _tn suffix or small size param)
        if 'shopee' in url:
            # Pattern: foo_tn.jpg → foo.jpg
            url = url.replace('_tn.', '.')
            # Pattern: @100w_100h_1e_2c_1l_1x → remove thumbnail params
            if '@' in url:
                url = url.split('@')[0]

        # Lazada CDN thumbnail → full size
        if 'lazada' in url or 'alicdn' in url:
            # Remove thumbnail size spec like _80x80.jpg
            import re
            url = re.sub(r'_\d+x\d+\.', '.', url)

        # Normalize protocol
        if url.startswith('//'):
            url = 'https:' + url

        result.append(url)

    return result
