#!/usr/bin/env python3
"""SPEAKER_NOTES.md from the \\talk{script}{sentence}{challenge}{answer} blocks of slides.tex (run by build.sh)."""
import re
from pathlib import Path
src = Path(__file__).with_name("slides.tex").read_text()
titles = re.findall(r"\\begin\{frame\}(?:\[plain\])?(?:\{([^}]*)\})?", src)
def args(pos, n=4):
    out = []
    for _ in range(n):
        assert src[pos] == "{"; depth = 0; i = pos
        while True:
            c = src[i]; depth += (c == "{") - (c == "}"); i += 1
            if depth == 0: break
        out.append(src[pos + 1:i - 1].strip()); pos = i
    return out
talks = [(m.start(), args(m.end())) for m in re.finditer(r"\\talk", src)]
frames = [(m.start(), (m.group(1) or "").strip()) for m in re.finditer(r"\\begin\{frame\}(?:\[plain\])?(?:\{([^}]*)\})?", src)]
lines = ["# Speaker notes (generated from slides.tex by extract_notes.py; edit the \\talk blocks, not this file)\n",
         "Each slide: a 15 to 40 s script, the one sentence to land, the likely expert challenge, the answer. Slide text is never read aloud.\n"]
k = 0
for pos, (script, sentence, challenge, answer) in talks:
    title = [t for p, t in frames if p < pos][-1] or ("HERO: FIVE SENSORS. ONE DECISION." if k == 0 else "TAKEAWAY (navy)")
    k += 1
    clean = lambda t: re.sub(r"\\[a-zA-Z]+\{([^}]*)\}", r"\1", t).replace("\\,", " ").replace("\\ ", " ").replace("~", " ").replace("$", "").replace("\\%", "%").replace("\\_", "_")
    lines += [f"\n## {k}. {clean(title)}\n", f"**Script.** {clean(script)}\n", f"**One sentence.** {clean(sentence)}\n", f"**Likely challenge.** {clean(challenge)}\n", f"**Answer.** {clean(answer)}\n"]
Path(__file__).with_name("SPEAKER_NOTES.md").write_text("\n".join(lines)); print("SPEAKER_NOTES.md:", k, "slides")
