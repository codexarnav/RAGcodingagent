import requests
import os
import time
import json
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()


GITHUB_TOKEN = os.getenv("GH_TOKEN")
if not GITHUB_TOKEN:
    raise ValueError("❌ GH_TOKEN not found! Make sure it's set in .env without quotes.")

HEADERS = {"Authorization": f"token {GITHUB_TOKEN}"}
OUTPUT_DIR = "data/github_issues"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def fetch_comments(comments_url):
    response = requests.get(comments_url, headers=HEADERS)
    if response.status_code == 200:
        return [comment["body"] for comment in response.json()]
    return []


def fetch_issues(owner, repo, state="closed", max_pages=5):
    print(f"🔎 Fetching issues from {owner}/{repo}... (state: {state})")
    issues = []

    for page in tqdm(range(1, max_pages + 1)):
        url = f"https://api.github.com/repos/{owner}/{repo}/issues"
        params = {
            "state": state,
            # "labels": ",".join(labels),  # <-- Commented for debug mode
            "per_page": 100,
            "page": page
        }

        response = requests.get(url, headers=HEADERS, params=params)
        if response.status_code != 200:
            print(f"❌ Error {response.status_code}: {response.text}")
            break

        page_data = response.json()
        real_issues = [i for i in page_data if "pull_request" not in i]

        for issue in real_issues:
            print(f"✅ #{issue['number']}: {issue['title']}")
            print("    Labels:", [label['name'] for label in issue['labels']])
        
        issues.extend(real_issues)
        if len(page_data) < 100:
            break
        time.sleep(1)

    return issues

def save_issues_to_json(owner, repo, filename=None):
    raw_issues = fetch_issues(owner, repo)
    cleaned = [
        {
            "title": issue.get("title", ""),
            "body": issue.get("body", ""),
            "labels": [l["name"] for l in issue.get("labels", [])],
            "url": issue.get("html_url", ""),
            "comments": fetch_comments(issue.get("comments_url", ""))
        }
        for issue in raw_issues
    ]

    filename = filename or f"{repo}_issues.json"
    filepath = os.path.join(OUTPUT_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2)

    print(f"\n💾 Saved {len(cleaned)} issues to {filepath}")


if __name__ == "__main__":
    save_issues_to_json("pallets", "flask")  # You can test others like 'scikit-learn'
