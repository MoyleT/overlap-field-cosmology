\# Overlap Field Cosmology (OFC)



\[!\[DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18626159.svg)](https://doi.org/10.5281/zenodo.18626159)

\[!\[License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



\*\*Numerical implementation of the cosmological projection of Cosmic Layer Superposition Theory (CLST) 4.7.\*\*



This repository contains the complete numerical framework for the \*\*overlap field\*\* cosmology model, including:

\- Calibration to ΛCDM (Ωₘ=0.3, H₀=1) via shooting method.

\- Systematic scan of the effective coupling parameter `Q ∈ \[-0.10, 0.15]`.

\- High‑precision `H(z)/H₀` interpolators for MCMC analysis.

\- Simulated Pantheon‑like supernova data and \*\*MCMC parameter estimation\*\* of `(Q, H₀)` using `emcee`.

\- \*\*One‑click reproduction\*\* of all four figures in the companion paper.



---



\## 🚀 Quick Start



\### 1. Install dependencies

```bash

pip install numpy scipy matplotlib emcee corner

```



\### 2. Reproduce all paper figures

```bash

python reproduce\_figures.py

```

This will generate `fig1\_calibration.pdf` – `fig4\_mcmc\_corner.pdf` in the current directory.



---



\## 📖 Documentation



\- \*\*Theory paper\*\*: \*Overlap Field Cosmology: Numerical Implementation and Parameter Constrainability Verification\* (arXiv:26XX.XXXXX, 2026)

\- \*\*Zenodo record\*\*: \[10.5281/zenodo.18626159](https://doi.org/10.5281/zenodo.18626159)

\- \*\*Full CLST theory\*\*: Ma, Y. (2026). \*Cosmic Layer Superposition Theory 4.7\* (in preparation)



---



\## 📁 Repository Structure



| File | Description |

|------|-------------|

| `reproduce\_figures.py` | One‑click figure reproduction |

| `OFC – Overlap Field Cosmology Simulator.py` | Core integrator and Q‑grid scan |

| `MCMC.py` | MCMC fitting with simulated supernova data |

| `requirements.txt` | Python dependencies |

| `figures/` | Output PDF figures |

| `docs/` | Additional documentation |



---



\## 📜 License



MIT License © 2026 Ma Yanliang



---



\## 📬 Contact



Ma Yanliang – Independent Researcher, Nanjing, China  

📧 1493704289@qq.com  

🔗 \[GitHub @MoyleT](https://github.com/MoyleT)

