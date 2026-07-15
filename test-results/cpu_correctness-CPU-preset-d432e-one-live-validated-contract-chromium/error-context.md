# Instructions

- Following Playwright test failed.
- Explain why, be concise, respect Playwright best practices.
- Provide a snippet of code with the fix, if possible.

# Test info

- Name: cpu_correctness.spec.ts >> CPU preset and custom URL settings form one live validated contract
- Location: tests_visual/cpu_correctness.spec.ts:292:5

# Error details

```
Error: page.evaluate: TypeError: Cannot read properties of null (reading 'value')
    at value (eval at evaluate (:303:30), <anonymous>:2:52)
    at eval (eval at evaluate (:303:30), <anonymous>:4:15)
    at UtilityScript.evaluate (<anonymous>:305:16)
    at UtilityScript.<anonymous> (<anonymous>:1:44)
```

# Page snapshot

```yaml
- generic [active] [ref=e1]:
  - generic:
    - text: DISTANCE 0 m
    - text: Renderer CPU splat
    - text: FPS —
    - text: Chunks 0
    - text: SVO nodes 0
    - text: Res scale —
    - text: GPU —
  - generic: …
  - generic [ref=e3] [cursor=pointer]:
    - heading "VACKROOMS" [level=1] [ref=e4]
    - paragraph [ref=e5]:
      - text: Click to enter — WASD move · mouse look · F flashlight · G drop flare · ESC release
      - text: "Touch: tap to enter — left thumb move · right thumb look · 🔥 drop flare"
    - button "Settings" [ref=e6]
```