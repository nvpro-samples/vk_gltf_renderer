#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Reproduce the spectral-radiance-to-luminance constants in shaders/sky_bruneton_io.h.slang.

Bruneton's model stores radiance at three wavelengths (680/550/440 nm) and converts to RGB
luminance with factors computed from the CIE colour-matching integral. It computes them **twice**,
with different spectral weightings:

    ComputeSpectralRadianceToLuminanceFactors(..., -3 /* lambda_power */, &sky_k_r, ...);
    ComputeSpectralRadianceToLuminanceFactors(...,  0 /* lambda_power */, &sun_k_r, ...);

The -3 compensates for the sky's blue-biased spectrum -- Rayleigh scattering goes as a steep
inverse power of wavelength -- so reconstructing luminance from three samples of *that* spectrum
needs red weighted up. The sun's spectrum has no such bias and needs the flat weighting. The two
sets carry the same units and differ only in their ratios, which is the colour: using the sky's set
for the sun over-boosts red by about 15% and renders sunlight several hundred Kelvin too warm.

Upstream computes the factors at runtime rather than publishing constants, and this port vendors
none of its tables. So this script reads them from upstream's own source -- the CIE 1931 table and
XYZ_TO_SRGB from `atmosphere/constants.h`, the ASTM G-173 solar spectrum from
`atmosphere/demo/demo.cc` -- and runs `ComputeSpectralRadianceToLuminanceFactors` from
`atmosphere/model.cc` line for line. The results are divided by MAX_LUMINOUS_EFFICACY, which is this
port's normalisation (see the header), and compared against the shipped constants.

Usage:
    python utils/sky_luminance_factors.py                 # fetch upstream from GitHub
    python utils/sky_luminance_factors.py --upstream DIR  # a local clone's atmosphere/ directory
"""

import argparse
import os
import re
import sys
import urllib.request

UPSTREAM_RAW = 'https://raw.githubusercontent.com/ebruneton/precomputed_atmospheric_scattering/master/atmosphere/'
LAMBDA_R, LAMBDA_G, LAMBDA_B = 680.0, 550.0, 440.0  # Model::kLambdaR/G/B
LAMBDA_MIN, LAMBDA_MAX = 360, 830

HEADER = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'shaders', 'sky_bruneton_io.h.slang')


def read_upstream(relpath, local_dir):
    if local_dir:
        with open(os.path.join(local_dir, relpath), encoding='utf-8') as f:
            return f.read()
    with urllib.request.urlopen(UPSTREAM_RAW + relpath) as r:
        return r.read().decode('utf-8')


def c_array(source, name):
    """The numbers of `... name[N] = { ... };` in a C++ source file."""
    m = re.search(re.escape(name) + r'\s*\[\s*\d*\s*\]\s*=\s*\{(.*?)\};', source, re.S)
    if not m:
        sys.exit(f'could not find {name} in upstream source')
    body = re.sub(r'//[^\n]*', '', m.group(1))
    return [float(v) for v in re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', body)]


def c_scalar(source, name):
    m = re.search(re.escape(name) + r'\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', source)
    if not m:
        sys.exit(f'could not find {name} in upstream source')
    return float(m.group(1))


def shipped(name):
    """Read a float3 #define out of the shader header, so this cannot drift from the source."""
    text = open(HEADER, encoding='utf-8').read()
    m = re.search(r'#define\s+' + name + r'\s+float3\(([^)]*)\)', text)
    if not m:
        return None
    return [float(v.strip().rstrip('Ff')) for v in m.group(1).split(',')]


def interpolate(wavelengths, values, wavelength):
    """model.cc Interpolate()."""
    if wavelength < wavelengths[0]:
        return values[0]
    for i in range(len(wavelengths) - 1):
        if wavelength < wavelengths[i + 1]:
            u = (wavelength - wavelengths[i]) / (wavelengths[i + 1] - wavelengths[i])
            return values[i] * (1.0 - u) + values[i + 1] * u
    return values[-1]


def cie_value(cie, wavelength, column):
    """model.cc CieColorMatchingFunctionTableValue()."""
    if wavelength <= LAMBDA_MIN or wavelength >= LAMBDA_MAX:
        return 0.0
    u = (wavelength - LAMBDA_MIN) / 5.0
    row = int(u)
    u -= row
    return cie[4 * row + column] * (1.0 - u) + cie[4 * (row + 1) + column] * u


def factors(cie, xyz2srgb, efficacy, wavelengths, solar, lambda_power):
    """model.cc ComputeSpectralRadianceToLuminanceFactors()."""
    lam = (LAMBDA_R, LAMBDA_G, LAMBDA_B)
    solar_ref = [interpolate(wavelengths, solar, l) for l in lam]
    k = [0.0, 0.0, 0.0]
    for l in range(LAMBDA_MIN, LAMBDA_MAX):
        xb, yb, zb = cie_value(cie, l, 1), cie_value(cie, l, 2), cie_value(cie, l, 3)
        bar = [xyz2srgb[3 * c] * xb + xyz2srgb[3 * c + 1] * yb + xyz2srgb[3 * c + 2] * zb for c in range(3)]
        irradiance = interpolate(wavelengths, solar, l)
        for c in range(3):
            k[c] += bar[c] * irradiance / solar_ref[c] * (l / lam[c]) ** lambda_power
    return [v * efficacy for v in k]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--upstream', help="local path to upstream's atmosphere/ directory")
    args = parser.parse_args()

    constants = read_upstream('constants.h', args.upstream)
    demo = read_upstream('demo/demo.cc', args.upstream)
    cie = c_array(constants, 'CIE_2_DEG_COLOR_MATCHING_FUNCTIONS')
    xyz2srgb = c_array(constants, 'XYZ_TO_SRGB')
    efficacy = c_scalar(constants, 'MAX_LUMINOUS_EFFICACY')
    solar = c_array(demo, 'kSolarIrradiance')
    wavelengths = [float(LAMBDA_MIN + 10 * i) for i in range(len(solar))]  # demo.cc: 360..830 step 10

    print(f'{"":26}{"r":>10} {"g":>10} {"b":>10}   ratios r:g:b')
    for label, name, power in (('sky (lambda_power = -3)', 'SKY_SPECTRAL_TO_LUMINANCE', -3.0),
                               ('sun (lambda_power =  0)', 'SUN_SPECTRAL_TO_LUMINANCE', 0.0)):
        k = [v / efficacy for v in factors(cie, xyz2srgb, efficacy, wavelengths, solar, power)]
        print(f'{label:26}{k[0]:10.4f} {k[1]:10.4f} {k[2]:10.4f}   {k[0]/k[1]:.3f} : 1 : {k[2]/k[1]:.3f}')
        have = shipped(name)
        if have:
            err = [abs(k[c] - have[c]) / have[c] * 100.0 for c in range(3)]
            print(f'{"  " + name:26}{have[0]:10.4f} {have[1]:10.4f} {have[2]:10.4f}   '
                  f'max error {max(err):.3f}%')


if __name__ == '__main__':
    main()
