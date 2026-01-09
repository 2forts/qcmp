# qcmp — Quantum Comparators Library (Qiskit)

`qcmp` is a **modular, extensible Python library for quantum comparators**, implemented in **Qiskit**.  
Its main goal is to provide a **unified software framework** to:

- implement *all* quantum comparators proposed in the literature,
- select a comparator by name,
- choose the bit-width `n`,
- provide classical inputs `(A, B)`,
- obtain the comparison result (`lt`, or later `lt/eq/gt`),
- and optionally evaluate **quantum cost metrics** in a consistent way.

This repository is designed to support **quantitative comparison and benchmarking** for review and research papers.

---

## ✨ Features

- 🔌 **Plugin-based architecture**: each comparator is a self-contained module
- 🔢 Arbitrary bit-width `n`
- ▶️ Simple CLI (`python -m qcmp`)
- 🧠 Clean separation between:
  - comparator logic
  - input encoding
  - simulation
  - metrics
- 📊 **Optional metrics module** (can be enabled or ignored)
- 🧪 Ready for unit testing and benchmarking
- 📚 Faithful implementations of **published quantum comparator circuits**

---

## 📦 Installation

Clone the repository and install in editable mode:

```bash
git clone https://github.com/<your-username>/qcmp.git
cd qcmp
pip install -e .
