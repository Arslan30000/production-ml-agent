"""Request a short-lived Airflow API token without putting a password in shell history."""

import getpass
import json
import os
import sys
from urllib.request import Request, urlopen


def main():
    base = os.environ.get("AIRFLOW_API_URL", "http://127.0.0.1:8080").rstrip("/")
    print("Airflow username: ", end="", file=sys.stderr, flush=True)
    user = input()
    password = getpass.getpass("Airflow password: ")
    payload = json.dumps({"username": user, "password": password}).encode()
    request = Request(base + "/auth/token", data=payload, method="POST",
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=15) as response:
        token = json.load(response)["access_token"]
    print(token)


if __name__ == "__main__":
    main()
