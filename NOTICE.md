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

The VLA Airy primary-beam geometry in `src/sl1mjax/beam.py` uses the public
CASA `vpmanager.setpbairy` defaults for the VLA: a 25 m dish, 2.5 m blockage,
and a maximum radius of 0.8 deg at 1 GHz. The optional RR/LL squint is the
standard feed offset of 0.06 of the Gaussian FWHM, opposite in the two hands.
The evaluation code is independent of CASA.
