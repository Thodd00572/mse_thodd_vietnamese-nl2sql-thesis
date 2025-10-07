---
trigger: always_on
---

Do not introduce code that is >70% similar to any function in src/ unless extracting shared helper is provided in the PR. If similarity found, suggest Extract Method or shared util and fail the check until resolved.