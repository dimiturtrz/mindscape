# fNIRS Simulator Papers: Code Repository & Integration Requirements

**Date**: 2026-07-09  
**Status**: partial  
**Papers**: arXiv 2605.30552 (2026, prioritized) | arXiv 2405.11242 (2024 precursor)

## TL;DR

Both papers claim open-source implementations but public code repositories are not discoverable (as of July 2026). The 2026 paper (Eastmond et al.) uses mesh-based Monte Carlo (MMC) with SPM12/Homer3/MNE-NIRS stack; contact eastmc@rpi.edu for code. The 2024 paper (Waks) uses Docker/Xarray/Python with Monte Carlo simulations, but repository is not publicly linked.

## Question

Where are the code repositories and what are the integration requirements (dependencies, Python version, API entry points, data requirements) for two recent fNIRS simulators: arXiv 2605.30552 (May 2026, Eastmond et al., RPI) and arXiv 2405.11242 (May 2024, Waks, Johns Hopkins APL)?

## Findings: arXiv 2605.30552 ("High-Fidelity 3D Simulator for Synthetic fNIRS Data Generation")

### Paper Details
- **Authors**: Condell Eastmond, Niels Bracher, Xavier Intes, Stefan T. Radev [S1]
- **Institution**: Rensselaer Polytechnic Institute (RPI), Center for Modeling, Simulation, and Imaging in Medicine [S6]
- **Submission**: May 28, 2026 [S1]
- **License**: CC BY 4.0 [S2]

### Code Availability & Contact
- **Repository Status**: Paper states "we provide an open-source implementation to support reproducibility" [S2], but no public GitHub URL is accessible as of July 2026 [S3]
- **Corresponding Author**: Condell Eastmond, eastmc@rpi.edu [S2]
- **Other Contacts**: Niels Bracher (brachn@rpi.edu), Xavier Intes (intesx@rpi.edu), Stefan T. Radev (radevs@rpi.edu) [S2]
- **Niels Bracher's GitHub** (niels-leif-bracher) shows 0 public repositories [S4]

### Core Implementation Details

**Monte Carlo Engine**: Mesh-based Monte Carlo (MMC) using the MMC toolbox [S1] with GPU parallelization [S2]
- GPU support via CUDA is inherent to MMC; see github.com/fangq/mmc (v2025.10) for reference implementation [S5]
- MMC achieves 420× speedup CPU→GPU via OpenCL/CUDA [S7]

**Mesh Generation**: ISO2Mesh referenced in broader fNIRS ecosystem [S8]
- Python binding available via pyiso2mesh (`pip install iso2mesh`) [S9]
- Generates tetrahedral meshes from volumetric images (brain atlases)

**Key Dependencies** (inferred from paper description) [S2]:
1. **SPM12 toolbox** – atlas coregistration (MATLAB dependency) [S2]
2. **Homer3** – signal denoising, preprocessing [S2]
3. **MNE-NIRS toolbox** – baseline/validation comparisons [S2]
4. **Automated Anatomical Labeling Atlas 3 (AAL3)** – brain parcellation [S2]
5. **Colin27 brain atlas** – anatomical reference [S2]

**Data Flow** (inferred):
- Input: Anatomical MRI (segmented tissue), optode montage definition (positions/wavelengths), target cortical activation patterns [S2]
- Process: Mesh generation → Monte Carlo photon transport → hemodynamic response injection → noise/artifact modeling [S2]
- Output: Synthetic HbO/HbR time series (NIRS data) with full-head spatiotemporal fidelity [S2]

**API Entry Points** (not explicitly documented in accessible content):
- Paper confirms simulator generates "hemodynamic responses, systemic physiology, and nonsystematic artifacts" across "configurable optode montages" [S1]
- Specific function names not quoted in available abstracts; contact author for MATLAB/Python API signatures

### Python vs MATLAB Status
- No explicit Python/MATLAB version requirements found in abstracts
- Dependencies (SPM12, Homer3) are MATLAB-based, suggesting hybrid MATLAB/Python architecture [S2]
- No pip-installable package found; likely requires source build

## Findings: arXiv 2405.11242 ("Advancing fNIRS Neuroimaging through Synthetic Data Generation and Machine Learning Applications")

### Paper Details
- **Author**: Eitan Waks [S10]
- **Institution**: Johns Hopkins University Applied Physics Laboratory (APL) [S6]
- **Submission**: May 18, 2024 [S10]
- **License**: CC BY 4.0 [S10]

### Code Availability & Contact
- **Repository Status**: Paper mentions "Docker and Xarray for standardized and reproducible data analysis" [S10] and "cloud-based infrastructure" [S10], but no public GitHub repository is linked [S11]
- **Author Contact**: Eitan Waks, Johns Hopkins APL [S6]
- No GitHub account directly tied to this work found [S11]

### Core Implementation Details

**Monte Carlo Engine**: Monte Carlo simulations with parametric head models [S10]
- Generates "comprehensive synthetic dataset reflecting a wide spectrum of conditions" [S10]
- No specific toolbox named; likely custom implementation or wrapping existing MMC/MCX

**Data Infrastructure** [S10]:
1. **Docker** – containerized reproducible environment (Dockerfile provided in paper) [S10]
2. **Xarray** – "Python library that provides support for labeled, multi-dimensional arrays" [S10]
3. **NetCDF format** – data storage (Xarray-native) [S10]

**Key Dependencies** (from paper description) [S10]:
- Python (version not specified)
- NumPy, SciPy
- Xarray
- Matplotlib, Seaborn
- h5py
- MNE, MNE-NIRS
- Jupyter / JupyterLab
- Custom Dockerfile based on Jupyter Docker Stacks [S10]

**Data Flow** [S10]:
- Input: Parametric head models, Monte Carlo simulation config
- Process: Monte Carlo photon transport → synthetic signal generation
- Output: Xarray-backed NetCDF datasets (HbO/HbR time series, labeled dimensions)
- Access: Jupyter notebooks for interactive analysis

**API Entry Points** (partially documented) [S10]:
- Paper references "JupyterLab Code" (Appendix A) with "Python scripts for database interaction and simulation configuration generation" [S10]
- No specific function signatures in available abstracts

### Python vs Docker Status
- Pure Python environment (no MATLAB dependency mentioned)
- Pip-installable: No—requires Docker image build or Dockerfile reproduction [S10]
- Installation: Users must build custom Docker image using provided Dockerfile [S10]
- Recent: Docker + Xarray pattern suggests modern Python stack

## Comparison & Integration Tractability

| Aspect | 2605.30552 (Eastmond et al.) | 2405.11242 (Waks) |
|--------|-----|-----|
| **Primary Engine** | MMC (mesh-based Monte Carlo) GPU-accelerated [S1] | Custom Monte Carlo + parametric models [S10] |
| **Language Stack** | MATLAB-dominant (SPM12, Homer3) + inference [S2] | Pure Python [S10] |
| **GPU Support** | Yes, via MMC's CUDA/OpenCL [S5] | Not documented [S10] |
| **Installation** | Source build from RPI (contact author) [S2] | Docker image build (Dockerfile in paper) [S10] |
| **Pip Installable** | Unknown; likely no [S2] | No, Docker-only [S10] |
| **Data Format** | Not specified; likely .mat (MATLAB) [S2] | NetCDF via Xarray [S10] |
| **Mesh Dependency** | ISO2Mesh required [S2] | Implicit (parametric models) [S10] |
| **Tissue Properties** | AAL3 atlas, Colin27 reference [S2] | Not documented [S10] |
| **External Data Downloads** | Manual (atlases, brain models) [S2] | Likely included in Docker or auto-fetched [S10] |

## Open Questions

- **Why no public GitHub?** Both papers claim open-source but repos are not discoverable. Possibilities: (1) papers are very recent (2605.30552 dated May 2026) and release is pending; (2) code lives on institutional servers (RPI, JHU-APL) with restricted access; (3) supplementary materials hosted on closed archives (OSF, Zenodo with restricted DOI).
- **2605.30552 API?** Function signatures, montage file format, HRF parameterization interface—not quoted in accessible abstracts. Requires full PDF or author contact.
- **2405.11242 Monte Carlo implementation?** Custom or wrapping fangq/mmc? Paper abstracts don't specify.
- **Tissue optical properties database?** Both claim physiologically accurate simulation but data source for μa/μs' not specified in abstracts.
- **Computational cost?** Runtime per synthetic recording, GPU memory requirements not documented in abstracts.

## Supporting Tools (Reference)

These tools underpin both simulators:

- **MMC (fangq/mmc)** [S5]: GPU-accelerated mesh-based Monte Carlo; Python pmmc module (`pip install pmmc`); v2025.10 with CUDA/OpenCL; GPL-3.0
- **ISO2Mesh / pyiso2mesh** [S9]: Mesh generation from volumetric images; Python via `pip install iso2mesh`; NumPy-based, no GPU; GPL-3.0
- **Brain2Mesh** [S8]: One-liner brain mesh generation; Python support as of July 2025 (`pip install iso2mesh`)
- **MC-simulation-fNIRS (Netaniel2)** [S12]: MATLAB-based reference using MCXLAB; open-source on GitHub

## Sources

- [S1] arXiv:2605.30552 abstract — https://arxiv.org/abs/2605.30552
- [S2] arXiv:2605.30552 HTML version, Appendix C (Code Availability) — https://arxiv.org/html/2605.30552v1
- [S3] GitHub search for "2605.30552" + "Eastmond" + "fNIRS simulator" (negative result, no public repo found) — Web search 2026-07-09
- [S4] GitHub profile niels-leif-bracher — https://github.com/niels-leif-bracher (0 public repositories as of 2026-07-09)
- [S5] GitHub fangq/mmc (Mesh-based Monte Carlo) — https://github.com/fangq/mmc (v2025.10, GPU-accelerated, Python pmmc module)
- [S6] Institutional affiliations: RPI Center for Modeling, Simulation, and Imaging in Medicine; Johns Hopkins APL Neural Interfaces — Web search 2026-07-09
- [S7] GPU-accelerated mesh-based Monte Carlo paper (biorxiv) — https://www.biorxiv.org/content/10.1101/815977v1.full
- [S8] Brain2Mesh project — http://www.mcx.space/brain2mesh/ (Python support as of 2025-07-30)
- [S9] GitHub NeuroJSON/pyiso2mesh — https://github.com/NeuroJSON/pyiso2mesh (Python Iso2Mesh binding, pip install iso2mesh)
- [S10] arXiv:2405.11242 abstract & HTML — https://arxiv.org/abs/2405.11242, https://arxiv.org/html/2405.11242v1
- [S11] GitHub search for "Eitan Waks" + "fNIRS" + "github" (negative result, no public repo linked) — Web search 2026-07-09
- [S12] GitHub Netaniel2/MC-simulation-fNIRS — https://github.com/Netaniel2/MC-simulation-fNIRS (MATLAB, MCXLAB-based)

---

**Next Steps for Integration**:
1. **Contact authors directly** for code access: eastmc@rpi.edu (2605.30552) or Eitan Waks (Johns Hopkins APL, Neural Interfaces)
2. **Check arXiv supplementary materials** directly for Zenodo/OSF links
3. **Evaluate MMC reference implementation** (fangq/mmc) as a fallback for Monte Carlo photon transport
4. **Assess MATLAB licensing** if pursuing 2605.30552 (SPM12, Homer3 require MATLAB)
