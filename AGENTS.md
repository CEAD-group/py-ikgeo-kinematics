# py-ikgeo-kinematics

Sibling of `py-opw-kinematics`; same shape, same conventions, same release
flow. Read `PLAN.md` before touching anything.

## Conventions

- Python orchestrates, Rust computes. Every inner loop is Rust; the Python
  layer converts arrays, applies the degrees flag and wraps scipy transforms.
- `ikgeo/` is a vendored BSD-3 crate (`ikgeo/NOTICE`). Keep it free of PyO3 and
  numpy so it can be used by other Rust consumers and tested with plain
  `cargo test`.
- API parity with py-opw-kinematics is a requirement, not a preference: method
  names, argument order, array shapes and NaN conventions are identical unless
  `PLAN.md` says otherwise. Downstream consumers select the backend per cell
  and must not branch on which one they got.
- Units: the model's own length unit throughout (consumers pass metres),
  degrees by default via the `degrees` flag, radians internally.
- Commits: Conventional Commit subjects, no `Co-Authored-By` trailer.
- Scratch files go in `./tmp/` (gitignored), never `/tmp/`.
- No em dashes. Complete sentences. Comments explain why, in one line. Write
  no documentation unless asked.

## Build and test

```bash
uv sync                                   # Python deps
uv run maturin develop --release          # build the extension in place
cargo test -p ikgeo --release             # Rust solver tests, no Python
uv run pytest                             # Python tests, incl. py-opw parity
uvx ruff check . && cargo fmt --check
```

Version is `0.0.0-dev` in `Cargo.toml`; CI stamps it from the `v*` tag. Any
PEP 517 build (`uv build`, `uv sync`, `pip install -e`) goes through
`build_backend/ikgeo_build_backend.py`, which stamps a `git describe` version
for the duration of the build: a clean tag gives `0.0.1`, anything past it
`0.0.2.dev<N>+g<sha>` (`.dirty` on an uncommitted tree), so a consumer's
`>=0.0.1` floor holds. A bare `maturin develop` bypasses the backend and still
reports `0.0.0`.
