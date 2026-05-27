import os
import re
import time
import threading
import requests
from dotenv import load_dotenv
from huggingface_hub import HfApi, hf_hub_download, list_repo_files

# Загружаем переменные из .env (токен)
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise ValueError("HF_TOKEN not found in .env file")

DATASET_REPO = "ExtraX0/YAna-SE_data"
RENDERER_ADD_URLS = "https://ExtraX0-yana-renderer.hf.space/add_urls"
PING_INTERVAL = 300          # 5 минут
COLLECT_INTERVAL = 300       # 5 минут

def get_existing_urls():
    """Возвращает множество URL уже обработанных страниц (из pages/)."""
    api = HfApi()
    files = list_repo_files(repo_id=DATASET_REPO, repo_type="dataset", token=HF_TOKEN)
    page_files = [f for f in files if f.startswith("pages/") and f.endswith(".json")]
    urls = set()
    for fname in page_files:
        try:
            path = hf_hub_download(repo_id=DATASET_REPO, filename=fname, repo_type="dataset", token=HF_TOKEN)
            with open(path, "r") as f:
                data = json.load(f)
                urls.add(data.get("url"))
        except Exception as e:
            print(f"Error reading {fname}: {e}")
    return urls

def extract_links_from_text(text: str) -> list:
    """Извлекает http/https ссылки из plain text."""
    urls = re.findall(r'https?://[^\s<>"\'()]+', text)
    cleaned = []
    for u in urls:
        u = u.rstrip('.,;:!?\'"')
        if u.startswith(('http://', 'https://')):
            cleaned.append(u)
    return list(set(cleaned))

def send_urls_to_renderer(urls):
    if not urls:
        return
    try:
        resp = requests.post(RENDERER_ADD_URLS, json={"urls": list(urls)}, timeout=30)
        print(f"Sent {len(urls)} URLs, response: {resp.json()}")
    except Exception as e:
        print(f"Failed to send URLs: {e}")

def ping_spaces():
    # Пингуем
    try:
        requests.get("https://ExtraX0-yana-renderer.hf.space/", timeout=10)
        print("[PING] Renderer OK")
    except Exception as e:
        print(f"[PING] Renderer error: {e}")
    try:
        requests.get("https://ExtraX0-yana-indexer.hf.space/health", timeout=10)
        print("[PING] Indexer OK")
    except Exception as e:
        print(f"[PING] Indexer error: {e}")

def collect_and_send():
    print(f"[{time.ctime()}] Collecting new links from pages...")
    existing = get_existing_urls()
    api = HfApi()
    files = list_repo_files(repo_id=DATASET_REPO, repo_type="dataset", token=HF_TOKEN)
    page_files = [f for f in files if f.startswith("pages/") and f.endswith(".json")]
    
    new_urls = set()
    for fname in page_files:
        path = hf_hub_download(repo_id=DATASET_REPO, filename=fname, repo_type="dataset", token=HF_TOKEN)
        with open(path, "r") as f:
            data = json.load(f)
        content = data.get("content", "")
        links = extract_links_from_text(content)
        for link in links:
            if link not in existing:
                new_urls.add(link)
    print(f"Found {len(new_urls)} new URLs")
    if new_urls:
        send_urls_to_renderer(new_urls)
    else:
        print("No new URLs found")

def pinger_loop():
    while True:
        ping_spaces()
        time.sleep(PING_INTERVAL)

def collector_loop():
    while True:
        try:
            collect_and_send()
        except Exception as e:
            print(f"Collector error: {e}")
        time.sleep(COLLECT_INTERVAL)

if __name__ == "__main__":
    threading.Thread(target=pinger_loop, daemon=True).start()
    collector_loop()
