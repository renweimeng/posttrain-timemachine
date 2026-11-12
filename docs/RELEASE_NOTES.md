# PostTrain Time Machine v0.2.0

Reward 涨了，模型真的学对了吗？PTM 把“看曲线”接到“查回答、改奖励、留测试”的完整流程。

## Download / 下载

- **Wheel**: `posttrain_timemachine-0.2.0-py3-none-any.whl` — bundled local web UI; ordinary users need no Node.js or GPU.
- **Source ZIP**: `posttrain-timemachine-0.2.0-source.zip` — source, examples, fixtures, documentation and prebuilt frontend.
- **SHA256SUMS** — SHA-256 checksums for both files.

Use 64-bit Python 3.11 or 3.12. See README for macOS, Linux and Windows PowerShell commands. No PyPI upload or native desktop installer is included.

## Included

- Timeline → completion inspection → declarative reward patch → rank/advantage diff.
- Four complete synthetic Reward Arena challenges, reviewed contracts, pytest/JUnit export and offline reports.
- L2 Token Signal Lens, mask-aware ratio/KL/surrogate inspection and complete-batch objective replay.
- Actual bounded CPU GPT-2 AdamW/SGD replay from numeric state, compared with recorded native after-step weights.
- Explicit-ID TRL reference capture, actual two-rank CPU L1 fixture, explicit-field verl and recorded multi-turn import.
- Persistent drafts, evidence links, analysis/report history, responsive layout and accessibility checks.

## Verification and scope

2026-11-12 local release check: 81 Python tests and 12 browser journeys passed, along with Ruff, mypy, ESLint, TypeScript and frontend build. The wheel is verified in an isolated environment without Torch. Cross-platform install smoke workflows cover Linux, macOS and Windows; their actual execution status is shown in GitHub Actions.

Native reference uses pinned TRL 0.24.0. GPU execution, generic trainer compatibility, native Windows deep capture and long-term model improvements are not claimed. The optional pinned training extra has known dependency advisories; see docs/SECURITY.md. Ordinary application dependencies are separate.

This is a local, single-user tool without login or multi-tenant cloud hosting. All synthetic examples are labeled. Fixed-answer reward edits and one-step replay are evidence for investigation, not predictions of future model quality.
