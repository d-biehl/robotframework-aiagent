# Examples

This folder contains extracted examples from the main README, split into individual Robot Framework suites for easier enrichment.

Notes

- These suites are illustrative; some use placeholder model names and rely on environment configuration (API keys, installed providers).
- The library is imported as `AIAgent`. Ensure the package is installed and importable in your environment.
- Some files demonstrate patterns (Planner/Executor, tool usage) and may reference keywords or tools you need to provide.
- Multimodal examples are model/provider dependent and may require additional wiring.

Suggested next steps

- Fill in concrete models you use in your environment.
- Add your own tools and allow-lists if you want agents to call Robot keywords.
- Add assertions around typed outputs for stable tests.

Index

- 01-quickstart.md
- 02-multi-agent-ping-pong.md
- 03-structured-classification.md
- 04-chain-of-responsibility.md
- 05-planner-executor-loop.md
- 06-author-reviewer.md
- 07-ensemble-arbitration.md
- 08-fallback-routing.md
- 09-execute-free-text-spec.md
- 10-per-step-model-switch.md
- 11-multimodal-attachment.md
- 12-config-by-example.md
- 13-structured-outputs-default.md
- 14-structured-outputs-override-per-call.md
- 15-history-helpers.md
- 16-per-call-model-settings.md
