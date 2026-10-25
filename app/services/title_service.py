import logging
from pathlib import Path
import re
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("clipforge.title")

MRBEAST_KEYWORD = "MrBeast"

# Minor words to keep lower-case during Title Casing (unless first or last word)
LOWERCASE_WORDS = {
    "a", "an", "the", "and", "but", "or", "for", "nor", "on", "at", "to",
    "from", "by", "over", "in", "of", "into", "with", "than", "as",
}


def parse_srt_text(srt_path: Path) -> str:
    """
    Extract plain dialogue text from an SRT subtitle file.
    Filters out numeric indices, timestamp lines, and formatting tags.
    """
    srt_path = Path(srt_path)
    if not srt_path.exists():
        return ""

    try:
        content = srt_path.read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        logger.warning(f"Failed to read SRT file {srt_path}: {e}")
        return ""

    lines = content.splitlines()
    dialogue_lines: List[str] = []

    for line in lines:
        line = line.strip()
        if not line or line.isdigit() or "-->" in line:
            continue
        # Strip HTML/ASS formatting tags like <i>, <b>, <font...>
        clean = re.sub(r"<[^>]+>", "", line).strip()
        if clean:
            dialogue_lines.append(clean)

    return " ".join(dialogue_lines).strip()


def format_title_casing(title: str) -> str:
    """
    Apply strategic title casing:
    - Preserves exact "MrBeast" and "MrBeast's" capitalization
    - Capitalizes major words (Nouns, Verbs, Adjectives, Adverbs)
    - Keeps minor prepositions and conjunctions lowercased
    """
    words = title.strip().split()
    if not words:
        return f"{MRBEAST_KEYWORD}'s Insane Challenge"

    cased: List[str] = []
    for i, w in enumerate(words):
        w_lower = w.lower()

        # Preserve exact MrBeast / MrBeast's capitalization
        if w_lower in {"mrbeast", "mrbeast's", "mrbeasts"}:
            if w_lower.endswith("'s"):
                cased.append(f"{MRBEAST_KEYWORD}'s")
            else:
                cased.append(MRBEAST_KEYWORD)
            continue

        clean_w = re.sub(r"[^\w]", "", w_lower)
        if i == 0 or i == len(words) - 1 or clean_w not in LOWERCASE_WORDS:
            # Preserve special acronyms like AI
            if clean_w == "ai":
                cased.append("AI")
            elif clean_w in {"tv", "hd", "mp4"}:
                cased.append(clean_w.upper())
            else:
                cased.append(w.capitalize())
        else:
            cased.append(w_lower)

    result = " ".join(cased)

    # Validate that exact keyword "MrBeast" is included
    if MRBEAST_KEYWORD not in result:
        result = f"{MRBEAST_KEYWORD} {result}"

    return result


def clean_title_length(title: str, max_chars: int = 70) -> str:
    """
    Ensure title stays concise and readable (target 40-70 characters when practical).
    Trims cleanly at word boundaries if too long, while ensuring 'MrBeast' is never cut.
    """
    title = re.sub(r"\s+", " ", title).strip()
    # Strip trailing punctuation
    title = title.rstrip(".,;:- ")

    if len(title) > max_chars:
        cutoff = title[:max_chars].rfind(" ")
        if cutoff > 35:
            trimmed = title[:cutoff].rstrip(".,;:- ")
            if MRBEAST_KEYWORD in trimmed:
                title = trimmed

    return title


def validate_mrbeast_title(title: str) -> bool:
    """
    Validate that the generated title meets all mandatory requirements:
    1. Contains the exact keyword "MrBeast"
    2. Is non-empty and has reasonable length (>= 25 characters)
    3. Does not contain hashtags or excessive exclamation marks.
    """
    if not title or not isinstance(title, str):
        return False
    if MRBEAST_KEYWORD not in title:
        return False
    if len(title.strip()) < 25:
        return False
    if "#" in title:
        return False
    if title.count("!") > 1:
        return False
    return True


def ensure_mrbeast_title(title: str, fallback_index: int = 1) -> str:
    """
    Guarantees that a title is valid, properly formatted, and includes 'MrBeast'.
    Corrects any missing keyword or invalid formatting.
    """
    if not title or not title.strip():
        return f"{MRBEAST_KEYWORD}'s Biggest Challenge #{fallback_index}"

    title = re.sub(r"[#@*~`]", "", title)
    title = re.sub(r"!+", "", title).strip()

    if MRBEAST_KEYWORD not in title:
        # Prepend MrBeast cleanly
        if title.lower().startswith("how ") or title.lower().startswith("why "):
            parts = title.split(" ", 1)
            title = f"{parts[0]} {MRBEAST_KEYWORD} {parts[1]}"
        elif title.lower().startswith("the "):
            title = f"{MRBEAST_KEYWORD}'s {title[4:]}"
        else:
            title = f"{MRBEAST_KEYWORD} {title}"

    return clean_title_length(format_title_casing(title))


def extract_content_signals(text: str) -> Dict[str, Any]:
    """
    Extract semantic signals, entities, amounts, time frames, and action types
    from speech or transcript dialogue.
    """
    lower = text.lower()

    # 1. Money & Prizes
    money_m = re.search(
        r"(\$\s*[\d,]+(?:\.\d+)?|\b\d+\s*(?:million|billion|thousand|dollars|bucks|k)\b)",
        lower,
    )
    money_str = ""
    if money_m:
        raw_m = money_m.group(0).strip()
        money_str = raw_m.upper().replace(" ", "")

    # 2. Quantities & Counts (e.g. 100 people, 24 hours, 60 seconds)
    qty_m = re.search(
        r"\b(\d{1,4})\s*(hours?|days?|seconds?|minutes?|people|players|fans|houses?|cities|cars?)\b",
        lower,
    )
    qty_str = qty_m.group(0) if qty_m else ""

    # 3. Detect prominent subject entities
    nouns: List[str] = []
    if "city" in lower or "cities" in lower:
        nouns.append("an Entire City")
    elif "house" in lower or "houses" in lower:
        nouns.append("Houses")
    elif "island" in lower:
        nouns.append("a Private Island")
    elif "bunker" in lower or "shelter" in lower:
        nouns.append("a Secret Bunker")
    elif "factory" in lower or "chocolate" in lower:
        nouns.append("a Giant Factory")
    elif "car" in lower or "cars" in lower or "jet" in lower or "plane" in lower or "vehicle" in lower:
        nouns.append("the World's Craziest Vehicle")
    elif "trap" in lower or "laser" in lower or "maze" in lower:
        nouns.append("the World's Hardest Maze")
    elif "studio" in lower or "shorts" in lower or "video" in lower or "ai" in lower:
        nouns.append("an Insane Video Studio")
    elif "hotel" in lower or "room" in lower:
        nouns.append("a $1,000,000 Hotel Room")
    elif "challenge" in lower:
        nouns.append("This Insane Challenge")

    # 4. Action themes
    has_build = any(w in lower for w in ["build", "built", "construct", "create", "make", "assemble", "craft"])
    has_give = any(w in lower for w in ["give", "giving", "gave", "won", "winner", "prize", "dollar", "cash", "million", "thousand", "pay", "spend", "bought"])
    has_survive = any(w in lower for w in ["survive", "survived", "escape", "trapped", "bunker", "island", "last", "stay", "floor", "danger", "room"])
    has_test = any(w in lower for w in ["test", "tested", "experiment", "try", "tried", "see if", "check"])
    has_speed = any(w in lower for w in ["second", "seconds", "minute", "minutes", "fast", "faster", "speed", "quick", "clock", "timer"])
    has_twist = any(w in lower for w in ["wrong", "unexpected", "twist", "crazy", "impossible", "intense", "insane", "shocking", "accident", "fail"])
    has_tech = any(w in lower for w in ["ai", "video", "shorts", "studio", "neural", "tracking", "crop", "software", "code"])

    return {
        "text": text,
        "money_str": money_str,
        "qty_str": qty_str,
        "nouns": nouns,
        "has_build": has_build,
        "has_give": has_give,
        "has_survive": has_survive,
        "has_test": has_test,
        "has_speed": has_speed,
        "has_twist": has_twist,
        "has_tech": has_tech,
    }


def generate_clip_title(
    clip_text: str = "",
    srt_path: Optional[Path] = None,
    clip_index: int = 1,
    total_clips: int = 1,
    used_titles: Optional[Set[str]] = None,
    source_title: str = "",
) -> str:
    """
    Generate an engaging, unique YouTube Shorts title for a clip based on its actual content.
    Mandatory requirement: Every title must contain the keyword 'MrBeast'.

    Content preference:
    1. Clip-specific .srt subtitle file
    2. Clip transcript text / segment speech
    3. Video metadata source title
    """
    if used_titles is None:
        used_titles = set()

    # 1. Content acquisition in order of preference
    raw_text = ""
    if srt_path and Path(srt_path).exists():
        raw_text = parse_srt_text(Path(srt_path))

    if not raw_text and clip_text:
        raw_text = clip_text.strip()

    if not raw_text and source_title:
        raw_text = source_title.strip()

    # Extract signals and keywords from content
    signals = extract_content_signals(raw_text)
    lower = raw_text.lower()
    money_str = signals["money_str"]
    qty_str = signals["qty_str"]
    nouns = signals["nouns"]

    candidates: List[str] = []

    # Category 1: Money / Giving Away / Spending / Prizes
    if signals["has_give"] or money_str:
        prize = money_str if money_str else ("Millions" if "million" in lower else "a Huge Prize")
        candidates.extend([
            f"{MRBEAST_KEYWORD} Gives Away {prize} in This Insane Challenge",
            f"{MRBEAST_KEYWORD}'s Most Expensive {prize} Giveaway Ever",
            f"{MRBEAST_KEYWORD} Surprises a Lucky Fan With {prize}",
            f"Why {MRBEAST_KEYWORD} Spent {prize} on This Challenge",
            f"{MRBEAST_KEYWORD} Puts {prize} on the Line for This Stunt",
        ])

    # Category 2: Building / Crafting / Constructing
    if signals["has_build"]:
        obj = nouns[0] if nouns else ("an Entire City" if "city" in lower else "Houses")
        candidates.extend([
            f"{MRBEAST_KEYWORD} Builds {obj} Faster Than You Can Imagine",
            f"{MRBEAST_KEYWORD} Builds an Entire {obj} for a Huge Challenge",
            f"{MRBEAST_KEYWORD}'s Most Expensive Project Ever",
            f"{MRBEAST_KEYWORD} Turns an Impossible Idea Into Reality",
            f"How {MRBEAST_KEYWORD} Built {obj} in Record Time",
            f"{MRBEAST_KEYWORD} Creates the Ultimate {obj}",
        ])

    # Category 3: Survival / Trapped / Escape / Endurance
    if signals["has_survive"] or "survive" in lower:
        loc = qty_str if qty_str else (nouns[0] if nouns else "100 Hours")
        candidates.extend([
            f"{MRBEAST_KEYWORD} Survives {loc} in the Craziest Challenge",
            f"{MRBEAST_KEYWORD}'s Wildest Survival Challenge Yet",
            f"{MRBEAST_KEYWORD} Trapped in Extreme Challenge for 24 Hours",
            f"{MRBEAST_KEYWORD} Escapes the World's Most Extreme Trap",
            f"{MRBEAST_KEYWORD}'s Survival Challenge Goes Completely Wrong",
            f"{MRBEAST_KEYWORD} Faces an Impossible Survival Challenge",
        ])

    # Category 4: High Pacing / Unexpected Turns / Twists
    if signals["has_twist"] or "wrong" in lower or "twist" in lower:
        candidates.extend([
            f"{MRBEAST_KEYWORD}'s Biggest Challenge Takes an Unexpected Turn",
            f"{MRBEAST_KEYWORD}'s Challenge Gets More Intense Every Second",
            f"{MRBEAST_KEYWORD} Pushes the Challenge to the Absolute Limit",
            f"Nobody Expected What {MRBEAST_KEYWORD} Did in the Final Round",
            f"{MRBEAST_KEYWORD}'s Wildest Stunt Almost Went Completely Wrong",
            f"{MRBEAST_KEYWORD}'s Secret Plan Finally Gets Revealed",
        ])

    # Category 5: Speed / Countdown / Urgency
    if signals["has_speed"] or "second" in lower or "fast" in lower:
        candidates.extend([
            f"{MRBEAST_KEYWORD} Beats the Clock in the Ultimate Challenge",
            f"{MRBEAST_KEYWORD}'s High-Speed Stunt Takes a Wild Turn",
            f"{MRBEAST_KEYWORD} Races Against Time to Win the Challenge",
            f"{MRBEAST_KEYWORD} Takes On a 60-Second Impossible Test",
        ])

    # Category 6: Tech / AI / Studio & Content Creation
    if signals["has_tech"] or "ai" in lower or "studio" in lower or "video" in lower:
        candidates.extend([
            f"{MRBEAST_KEYWORD} Tests Crazy New AI Video Technology",
            f"{MRBEAST_KEYWORD} Builds an Insane Studio for Viral Shorts",
            f"{MRBEAST_KEYWORD} Reveals How to Create Viral Shorts Fast",
            f"{MRBEAST_KEYWORD}'s Secret Setup for Making Viral Videos",
            f"{MRBEAST_KEYWORD} Turns Simple Footage Into a Viral Masterpiece",
        ])

    # Category 7: General High-Energy MrBeast Style Templates
    general_pool = [
        f"{MRBEAST_KEYWORD}'s Biggest Challenge Takes an Unexpected Turn",
        f"{MRBEAST_KEYWORD} Builds an Entire City for a Huge Challenge",
        f"{MRBEAST_KEYWORD} Gives Away Millions in This Insane Challenge",
        f"{MRBEAST_KEYWORD} Builds Houses Faster Than You Can Imagine",
        f"{MRBEAST_KEYWORD}'s Most Expensive Project Ever",
        f"{MRBEAST_KEYWORD} Turns an Impossible Idea Into Reality",
        f"{MRBEAST_KEYWORD}'s Challenge Gets More Intense Every Second",
        f"Inside {MRBEAST_KEYWORD}'s Most Extreme Challenge Yet",
        f"{MRBEAST_KEYWORD} Takes On the World's Hardest Challenge",
        f"{MRBEAST_KEYWORD}'s Final Challenge Changes Everything",
    ]
    candidates.extend(general_pool)

    # 4. Selection loop: pick the best candidate that is unused and meets requirements
    chosen_title = None

    for cand in candidates:
        formatted = format_title_casing(cand)
        formatted = clean_title_length(formatted)

        if validate_mrbeast_title(formatted) and formatted not in used_titles:
            chosen_title = formatted
            break

    # If all candidate strings are already used in this job, generate a unique variation
    if not chosen_title:
        base = candidates[(clip_index - 1) % len(candidates)]
        varied = f"{base} (Part {clip_index})"
        formatted = clean_title_length(format_title_casing(varied))
        if formatted in used_titles:
            formatted = f"{clean_title_length(format_title_casing(base))} #{clip_index}"
        chosen_title = formatted

    # Fallback safety enforcement
    chosen_title = ensure_mrbeast_title(chosen_title, fallback_index=clip_index)

    used_titles.add(chosen_title)
    return chosen_title
