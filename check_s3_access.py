#!/usr/bin/env python3
"""
check_s3_access.py
===============================================================================
Point 6 (pathway-level microbiome) hit a real blocker: the HUMAnN pathway
data exists only for oral_microbiome (not gut_microbiome), and its pointer
columns hold S3 URIs sharded one file per sample -- never downloaded locally,
unlike the pre-aggregated MetaPhlAn species table this project has used
throughout.

This script answers, in under a few seconds, whether pursuing it further is
even possible:
  1. Is boto3 installed and can it reach AWS at all?
  2. Do we have credentials with read access to this specific bucket?
  3. Can we HEAD (not download) one real object from the pointer list?
  4. Is there any local mirror of oral_microbiome/10k/humann/ anywhere
     under DATA_ROOT that the earlier rglob fallback might have missed?

    conda activate pheno
    cd ~/studies/HPP_Prototype
    python check_s3_access.py 2>&1 | tee check_s3_access.log
===============================================================================
"""
from __future__ import annotations

import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("HPP_DATA_ROOT", Path.home() / "studies" / "hpp_datasets"))
BUCKET = "pheno-big-data-v1.2-prod-eu-west-1"
SAMPLE_KEY = ("oral_microbiome/10k/humann/pathways/abundance/"
             "humann_pathway_abundance_pathway_level_parquet__10k_2019372998__20190311_0.parquet")


def log(m=""):
    print(m, flush=True)


def section(t):
    log("\n" + "=" * 78)
    log(f"  {t}")
    log("=" * 78)


def main():
    section("1. Is boto3 available?")
    try:
        import boto3
        from botocore.config import Config
        # THE BUG THIS FIXES: boto3's default timeouts are long (tens of
        # seconds per attempt, with retries), so on an instance with no route
        # to S3 at all -- common for a private subnet without a VPC S3
        # endpoint -- a plain client() call hangs for minutes before ever
        # producing an error. These short, explicit timeouts make "no network
        # path" fail fast instead of looking indistinguishable from "still
        # working".
        FAST = Config(connect_timeout=5, read_timeout=5,
                      retries={"max_attempts": 1})
        log(f"  boto3 version: {boto3.__version__}")
    except ImportError:
        log("  boto3 NOT INSTALLED. `pip install boto3` would be needed before")
        log("  any S3 access is possible from this environment.")
        boto3 = None

    if boto3:
        section("1b. Are credentials even configured? (checked before any network call)")
        try:
            session = boto3.Session()
            creds = session.get_credentials()
            if creds is None:
                log("  NO CREDENTIALS FOUND (no ~/.aws/credentials, no env vars, no "
                    "instance metadata role). S3 access is not possible until "
                    "credentials or an IAM role are attached to this instance.")
            else:
                frozen = creds.get_frozen_credentials()
                log(f"  credentials found, access key starts with: "
                    f"{frozen.access_key[:4]}...  (method: {creds.method})")
        except Exception as exc:
            log(f"  credential check failed: {exc}")

        section("2. Do we have credentials, and can we reach this bucket? (5s timeout)")
        try:
            s3 = boto3.client("s3", config=FAST)
            resp = s3.list_objects_v2(
                Bucket=BUCKET,
                Prefix="oral_microbiome/10k/humann/pathways/abundance/",
                MaxKeys=5,
            )
            n = resp.get("KeyCount", 0)
            log(f"  list_objects_v2 succeeded. KeyCount in first page: {n}")
            for obj in resp.get("Contents", []):
                log(f"    {obj['Key']}  ({obj['Size']/1e3:.1f} KB)")
            if "IsTruncated" in resp:
                log(f"  IsTruncated: {resp['IsTruncated']} "
                    f"(there are more than {n} objects under this prefix)")
        except Exception as exc:
            log(f"  FAILED: {type(exc).__name__}: {exc}")
            log("  This likely means no AWS credentials are configured, or this")
            log("  instance's IAM role does not have read access to this bucket.")

        section("3. HEAD one specific object from the real pointer list")
        try:
            s3 = boto3.client("s3", config=FAST)
            head = s3.head_object(Bucket=BUCKET, Key=SAMPLE_KEY)
            log(f"  HEAD succeeded: {head['ContentLength']/1e3:.1f} KB, "
                f"last modified {head['LastModified']}")
            log("  -> S3 access to this exact data is CONFIRMED WORKING.")
        except Exception as exc:
            log(f"  FAILED: {type(exc).__name__}: {exc}")

        section("4. How many total objects exist under the pathway-abundance prefix?")
        try:
            s3 = boto3.client("s3", config=FAST)
            paginator = s3.get_paginator("list_objects_v2")
            total = 0
            total_bytes = 0
            for page in paginator.paginate(
                Bucket=BUCKET,
                Prefix="oral_microbiome/10k/humann/pathways/abundance/",
            ):
                total += page.get("KeyCount", 0)
                for obj in page.get("Contents", []):
                    total_bytes += obj["Size"]
                if total > 20000:   # safety cap, this is just a count estimate
                    log(f"  ...stopped counting at {total:,} (still more remain)")
                    break
            log(f"  total objects counted: {total:,}")
            log(f"  total size so far: {total_bytes/1e9:.2f} GB")
            log(f"  average object size: {total_bytes/max(total,1)/1e3:.1f} KB")
        except Exception as exc:
            log(f"  FAILED: {type(exc).__name__}: {exc}")

    section("5. Any local mirror already on disk?")
    hits = list(DATA_ROOT.rglob("*humann*pathway*"))
    log(f"  files/dirs matching *humann*pathway* under {DATA_ROOT}: {len(hits)}")
    for h in hits[:20]:
        log(f"    {h.relative_to(DATA_ROOT)}")
    if not hits:
        log("  NONE. The pathway data is not mirrored locally in any form.")

    section("DONE")
    log("  Decision point:")
    log("  - If section 3 succeeded: S3 access works, and downloading/aggregating")
    log("    ~1,800 small pathway files (one per repeat-visit participant's visit)")
    log("    is feasible, though it is real new engineering, not a quick add-on.")
    log("  - If section 3 failed (no credentials/role): pathway-level analysis is")
    log("    blocked at the infrastructure level, not a coding problem, and is")
    log("    worth reporting to Adinath as such rather than spending more time on it.")
    log("  - Independent of S3 access: this data is ORAL, not GUT, microbiome.")
    log("    Even if fully accessible, it answers a different question than the")
    log("    paper's stated scope and should be framed as a separate follow-up,")
    log("    not folded into this paper's Point 6.")


if __name__ == "__main__":
    main()
