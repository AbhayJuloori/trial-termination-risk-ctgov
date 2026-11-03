"""Page through the ClinicalTrials.gov API v2 and save the raw study JSON.

Only the protocol modules the pipeline reads are requested (no results
section, no free-text descriptions), which cuts the download from about
10 GB to under 2 GB. Pages are written as gzipped JSON lines, one study per
line, so a failed run resumes from the last saved page.

    python scripts/download_studies.py
"""

from __future__ import annotations

import gzip
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "studies"
API = "https://clinicaltrials.gov/api/v2/studies"
FIELDS = ",".join([
    "protocolSection.identificationModule.nctId",
    "protocolSection.identificationModule.organization",
    "protocolSection.statusModule",
    "protocolSection.sponsorCollaboratorsModule",
    "protocolSection.oversightModule",
    "protocolSection.conditionsModule",
    "protocolSection.designModule",
    "protocolSection.armsInterventionsModule",
    "protocolSection.outcomesModule",
    "protocolSection.eligibilityModule",
    "protocolSection.contactsLocationsModule.locations.country",
    "protocolSection.contactsLocationsModule.locations.facility",
    "derivedSection.conditionBrowseModule.ancestors",
    "derivedSection.miscInfoModule.versionHolder",
])


def fetch(token: str | None) -> dict:
    params = {"pageSize": 1000, "fields": FIELDS, "format": "json"}
    if token:
        params["pageToken"] = token
    url = API + "?" + urllib.parse.urlencode(params)
    for attempt in range(6):
        try:
            with urllib.request.urlopen(url, timeout=120) as r:
                return json.loads(r.read())
        except Exception as e:  # 429 or transient network errors
            wait = 10 * (attempt + 1)
            print(f"retry in {wait}s: {e}", flush=True)
            time.sleep(wait)
    raise RuntimeError("API kept failing")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    state = OUT / "_next_token.json"
    page, token = 0, None
    if state.exists():
        s = json.loads(state.read_text())
        page, token = s["page"], s["token"]
        if token is None:
            print("already complete")
            return 0
    while True:
        data = fetch(token)
        with gzip.open(OUT / f"page_{page:05d}.jsonl.gz", "wt") as fh:
            for st in data["studies"]:
                fh.write(json.dumps(st) + "\n")
        token = data.get("nextPageToken")
        page += 1
        state.write_text(json.dumps({"page": page, "token": token}))
        if page % 20 == 0:
            print(f"{page} pages", flush=True)
        if not token:
            break
        time.sleep(0.5)
    print(f"done: {page} pages")
    return 0


if __name__ == "__main__":
    sys.exit(main())
