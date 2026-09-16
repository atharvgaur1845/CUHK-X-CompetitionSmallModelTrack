# Most Popular Award — public-facing copy (form due 18 Sept 23:59 UTC)

**Project title:** Five Sensors, One Model: Multimodal Activity Recognition Under a 100 MB Constraint

**Team / track:** AGX — Small Model Track. Affiliation, country, members, photo: TODO-TEAM.

**Introduction (95 words):** People are recognisable in ordinary video, so this challenge asks
whether a machine can tell forty everyday activities apart from sensors that preserve privacy: an
infrared and a depth camera, a thermal camera, body-joint tracking, wrist-worn motion sensors and
radar. Our system lets five small specialists each read the stream they are best at — the whole
body, the hands, the heat of objects, the skeleton, the wrist motion — and combines their votes in a
principled way, then uses the natural order of household routines to resolve doubts. All of it fits
in under 100 MB and runs on people it has never seen.

**Highlights:**
1. A "crop-size rule": higher resolution helps the body view and hurts the wrist view — measured, not guessed.
2. The thermal camera, useless at low resolution, becomes a key sensor at full frame.
3. Fusing five small models makes accuracy fairer across people, narrowing the gap between the
   easiest and hardest subject from 28 to 21 points.

**Link:** https://github.com/atharvgaur1845/CUHK-X-CompetitionSmallModelTrack

**Visual identity:** five sensor glyphs (IR, depth, thermal, skeleton, wrist IMU) converging into one
96 MB chip; palette from the report figures (blue/grey/orange).
