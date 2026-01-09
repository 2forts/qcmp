# qcmp — Quantum Comparators Library (Qiskit)

`qcmp` is a **modular and extensible Python library for quantum comparators**, implemented in **Qiskit**.  
Its purpose is to provide a **unified software framework** to implement, test, and compare *all* quantum comparator circuits proposed in the literature.

The library is designed to support **research, benchmarking, and review papers** in quantum arithmetic and comparison circuits.

---

## ✨ Features

- 🔌 **Plugin-based architecture**: each comparator is an independent module
- 🔢 Arbitrary bit-width `n`
- ▶️ Simple command-line interface (`python -m qcmp`)
- 🧠 Clean separation between:
  - comparator logic
  - input encoding
  - simulation
  - metrics
- 📊 **Optional metrics module** (can be enabled or ignored)
- 🧪 Ready for unit testing and benchmarking
- 📚 Faithful implementations of published quantum comparator circuits

---

## 📦 Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/<your-username>/qcmp.git
cd qcmp
pip install -e .
```

Optional (for circuit visualization with matplotlib):

```bash
pip install pylatexenc
```

---

## 🚀 Quick Start

### List available comparators

```bash
python -m qcmp list
```

Example output:
```
lt_orts_tand
```

---

### Run a comparator

```bash
python -m qcmp run --cmp lt_orts_tand --n 4 --A 1 --B 2
```

Example output:
```json
{
  "cmp": "lt_orts_tand",
  "n": 4,
  "A": 1,
  "B": 2,
  "decoded": { "lt": 1 },
  "counts": {
    "0 000": 251,
    "0 001": 256,
    "0 010": 260,
    "0 011": 257
  },
  "metrics": null
}
```

---

## Project Structure

The repository is organized as follows:

- `pyproject.toml` — package metadata and dependencies
- `README.md` — (this file)
- `qcmp/`
  - `__init__.py`
  - `__main__.py` — CLI entrypoint (`python -m qcmp ...`)
  - `encoding.py` — classical integer → basis-state encoding (X gates)
  - `registry.py` — comparator registration + lookup
  - `runner.py` — execution pipeline (prep inputs, transpile, simulate, decode, optional metrics)
  - `result.py` — `RunResult` dataclass
  - `comparators/`
    - `__init__.py` — registers available comparator plugins
    - `base.py` — comparator interface (`ComparatorMeta`, `BuildOutput`)
    - `lt_orts_tand.py` — Orts et al. (2021) LT comparator (temporary logical-AND based)
  - `metrics/`
    - `__init__.py` — metrics registry + implementations (current repo keeps metrics here)
- `tests/`
  - `test_smoke.py` — minimal smoke test

This structure allows new comparators and metrics to be added with minimal changes to the core code.

---

## Implemented Comparator

lt_orts_tand

Reference:
F. Orts et al.,
Optimal fault-tolerant quantum comparators for image binarization,
The Journal of Supercomputing, 2021.

## 📄 License

MIT License
