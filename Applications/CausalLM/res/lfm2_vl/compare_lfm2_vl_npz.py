#!/usr/bin/env python3
"""Compare tensor dumps produced during the LFM2-VL port."""

import argparse
import sys

import numpy as np


def _as_float(array):
    if array.dtype.kind in {"b", "i", "u"}:
        return array.astype(np.float64)
    return array.astype(np.float64, copy=False)


def compare_key(key, ref, cand, rtol, atol):
    ref_arr = ref[key]
    cand_arr = cand[key]

    if ref_arr.shape != cand_arr.shape:
        return False, f"shape mismatch: ref={ref_arr.shape}, cand={cand_arr.shape}"

    if ref_arr.dtype.kind in {"U", "S", "O"}:
        ok = np.array_equal(ref_arr, cand_arr)
        return ok, "exact match" if ok else "string/object mismatch"

    if ref_arr.dtype.kind in {"b", "i", "u"} and cand_arr.dtype.kind in {"b", "i", "u"}:
        ok = np.array_equal(ref_arr, cand_arr)
        if ok:
            return True, "exact match"
        diff_count = np.count_nonzero(ref_arr != cand_arr)
        return False, f"{diff_count} elements differ"

    ref_f = _as_float(ref_arr)
    cand_f = _as_float(cand_arr)
    diff = np.abs(ref_f - cand_f)
    max_abs = float(diff.max()) if diff.size else 0.0
    denom = np.maximum(np.abs(ref_f), 1e-12)
    max_rel = float((diff / denom).max()) if diff.size else 0.0
    ok = np.allclose(ref_f, cand_f, rtol=rtol, atol=atol, equal_nan=True)
    return ok, f"max_abs={max_abs:.8g}, max_rel={max_rel:.8g}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("reference")
    parser.add_argument("candidate")
    parser.add_argument("--keys", nargs="*", default=None)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()

    ref = np.load(args.reference, allow_pickle=False)
    cand = np.load(args.candidate, allow_pickle=False)

    keys = args.keys or sorted(set(ref.files) & set(cand.files))
    missing_ref = [key for key in keys if key not in ref.files]
    missing_cand = [key for key in keys if key not in cand.files]
    if missing_ref or missing_cand:
        if missing_ref:
            print(f"Missing in reference: {', '.join(missing_ref)}")
        if missing_cand:
            print(f"Missing in candidate: {', '.join(missing_cand)}")
        return 2

    failed = False
    for key in keys:
        ok, message = compare_key(key, ref, cand, args.rtol, args.atol)
        status = "OK" if ok else "FAIL"
        print(f"{status:4} {key}: {message}")
        failed = failed or not ok

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
