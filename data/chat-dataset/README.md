# Chat Dataset (generated)

- Source repo: https://github.com/ArjunDivecha/Factor-Timing-Value
- Source commit (SHA): 808ae10182a50623216c23c9dc964a2a397de324
- Counts: total=508, train=458, valid=50

Command used:
```
gh-chat-dataset --repo https://github.com/ArjunDivecha/Factor-Timing-Value --out out/factor \
  --md-max-questions-per-section 4 --md-window-tokens 800 \
  --py-chunking --py-chunk-max 5 --py-chunk-min-lines 6 \
  --max-tokens 4096 --min-tokens 48 --file-cap 15
```
