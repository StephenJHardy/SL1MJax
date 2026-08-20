# Provenance

SL1MJax is a new JAX implementation of differentiable radio-interferometric
sky and instrument modelling. Its architecture follows the revised development
plan and the conceptual direction explored by the MIT-licensed SL1MML research
prototype.

The historical SL1M repository contains GPL-2 code. No implementation code is
copied from it. Published measurement equations and independently derived
analytic tests are used instead.

SL1MJax is distributed under the BSD 3-Clause License.

The versioned VLA gain-curve coefficient catalog in
`src/sl1mjax/data/vla_gain_curves.json` is derived from NRAO's public CASA
`casa-data/nrao/VLA/GainCurves` table at commit
`65a746f9e6661da9a52484c47b5ad159b5c50234`. The coefficients are observatory
calibration data; the SL1MJax selection and evaluation implementation is
independent.
