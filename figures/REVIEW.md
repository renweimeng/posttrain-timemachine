# Mechanism diagram review

Rendered with Mermaid CLI 12.2.0. Both raw .mmd files match their Markdown wrappers and render successfully as PNG. Local Puppeteer configuration is excluded from the repository; rendering uses an installed browser.

## Workflow — 9/10, ACCEPT

Five stages run left to right: capture, inspect, patch, recompute, reviewed tests/reports. The dashed feedback arrow returns to capture and explicitly requires retraining to validate. All arrows and stage labels were visually checked against the implementation. White background, teal evidence stages and violet computation stages remain readable. The compact feedback label is the remaining cosmetic limitation.

## Evidence levels — 9/10, ACCEPT

Answers/rewards feed rank/advantage analysis; token tensors feed objective replay; both connect to reviewed tests and reports. The separate numeric checkpoint row feeds a real CPU step and a native weight comparison. No arrow implies that ordinary reports contain model weights or that replay predicts long-term quality. All labels and endpoints are visible; detailed data prerequisites are explained in README. Sources use short labels to avoid multiline clipping in the renderer.

The README hero is an authored SVG illustration, and the UI PNGs come from the local browser test suite. Synthetic and actual native-training evidence are explicitly labeled in the README captions.
