# velesdb-common

Shared utilities for [VelesDB](https://github.com/cyberlife-coder/VelesDB) Python integrations.

> **This package is not a public API.** End users should install and import from
> [`langchain-velesdb`](https://pypi.org/project/langchain-velesdb/) or
> [`llama-index-vector-stores-velesdb`](https://pypi.org/project/llama-index-vector-stores-velesdb/)
> directly.

## What it provides

`velesdb-common` centralizes code shared by both integration packages:

- **Security validators** — input sanitization for collection names, dimensions, queries, URLs
- **ID generation** — deterministic hashing and sequential ID counters
- **Graph helpers** — REST payload builders and native graph bindings
- **Memory formatting** — procedural memory result formatting

## License

MIT License (this integration). See [LICENSE](https://github.com/cyberlife-coder/VelesDB/blob/main/integrations/common/LICENSE) for details.

VelesDB Core itself is licensed under the [VelesDB Core License 1.0](https://github.com/cyberlife-coder/VelesDB/blob/main/LICENSE) (based on ELv2).
