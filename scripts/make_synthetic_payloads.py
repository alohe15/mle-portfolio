#!/usr/bin/env python3
"""Write a deterministic synthetic /predict payload file.

Uses only the standard library. Re-running this script rewrites the same bytes.
Values are invented. Nothing is read from data/.
"""

import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent.parent / "tests" / "load" / "payloads.json"

PRODUCTS = ("W", "C", "H", "R", "S")
CARD_BRANDS = ("visa", "mastercard", "discover", "american express")
CARD_TYPES = ("credit", "debit")
DOMAINS = ("example.com", "example.org", "example.net", "mail.example")
DEVICES = (("desktop", "GenericDesktop"), ("mobile", "GenericPhone"))
LARGE_IDS = {7, 16, 23}


def build_rows() -> list[dict]:
    rng = random.Random(42)
    rows = []
    for index in range(1, 25):
        if index in LARGE_IDS:
            amount = round(rng.uniform(1500, 3000), 2)
        else:
            amount = round(rng.uniform(5, 800), 2)
        device_type, device_info = DEVICES[index % 2]
        request = {
            "TransactionAmt": amount,
            "ProductCD": PRODUCTS[(index - 1) % len(PRODUCTS)],
            "card1": float(1000 + rng.randrange(18000)),
            "card2": float(100 + rng.randrange(500)),
            "card3": float(100 + rng.randrange(200)),
            "card4": CARD_BRANDS[rng.randrange(len(CARD_BRANDS))],
            "card5": float(100 + rng.randrange(200)),
            "card6": CARD_TYPES[index % 2],
            "addr1": float(100 + rng.randrange(400)),
            "addr2": float(10 + rng.randrange(90)),
            "P_emaildomain": DOMAINS[rng.randrange(len(DOMAINS))],
            "R_emaildomain": DOMAINS[rng.randrange(len(DOMAINS))],
            "dist1": float(rng.randrange(5000)),
            "TransactionDT": float(86400 * rng.randrange(1, 120)),
            "D1": float(rng.randrange(600)),
            "C1": float(rng.randrange(50)),
            "DeviceType": device_type,
            "DeviceInfo": f"{device_info}-{index:02d}",
        }
        rows.append({"example_id": f"syn-{index:03d}", "request": request})
    return rows


def main() -> None:
    text = json.dumps(build_rows(), indent=2, sort_keys=True) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(text)


if __name__ == "__main__":
    main()
