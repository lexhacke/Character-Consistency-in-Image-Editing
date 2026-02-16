"""
Local-first dataset for the scalable pipeline.

Source images loaded from disk (no Flickr). Only Apple CDN for edited images.
"""
import PIL.Image
import json
import requests
import io
import time
import asyncio
import aiohttp
from pathlib import Path


class LocalPicobananaDataset:
    def __init__(self, source_dir, start_index=0, n=200_000, return_img=True):
        self.source_dir = Path(source_dir)
        self.n = n
        self.return_img = return_img
        self.start_index = start_index

        print("Fetching JSONL index...")
        r = requests.get('https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb/jsonl/sft.jsonl')
        all_lines = [json.loads(line) for line in r.text.splitlines() if line.strip()]
        self.raw_data = all_lines[self.start_index : self.start_index + self.n]

        self.edit_types = set()
        self.data = []
        self.fails = 0
        self.local_hits = 0
        self.local_misses = 0

    async def prepare_data(self):
        print(f"Checking {len(self.raw_data)} items...")
        self.data = await self._filter_valid(self.raw_data)
        print(f"Retained {len(self.data)} valid items, {self.fails} fails. "
              f"Local: {self.local_hits} hits, {self.local_misses} misses")

    async def _filter_valid(self, raw_data):
        valid = []
        sem = asyncio.Semaphore(50)
        async with aiohttp.ClientSession() as session:
            tasks = [self._check_item(sem, session, item) for item in raw_data]
            from tqdm.asyncio import tqdm
            for result in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
                item = await result
                if item:
                    valid.append(item)
        return valid

    async def _check_item(self, sem, session, item):
        async with sem:
            try:
                img_id = item['open_image_input_url'].split('/')[-1]
                local_path = self.source_dir / img_id

                if not local_path.exists():
                    self.local_misses += 1
                    self.fails += 1
                    return None

                self.local_hits += 1

                edit_url = "https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb/" + item['output_image']
                async with session.head(edit_url, timeout=5) as resp:
                    if resp.status != 200:
                        self.fails += 1
                        return None

                self.edit_types.add(item['edit_type'])
                return item
            except:
                self.fails += 1
                return None

    def __getitem__(self, index):
        item = self.data[index]
        prompt = item['text']
        edit_type = item['edit_type']
        img_id = item['open_image_input_url'].split('/')[-1]
        local_path = self.source_dir / img_id
        edit_url = "https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb/" + item['output_image']

        if self.return_img:
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    og_img = PIL.Image.open(local_path).convert("RGB")

                    resp_edit = requests.get(edit_url, timeout=30)
                    if resp_edit.status_code == 429:
                        raise requests.exceptions.HTTPError("429 Too Many Requests", response=resp_edit)
                    resp_edit.raise_for_status()
                    edit_img = PIL.Image.open(io.BytesIO(resp_edit.content)).convert("RGB")

                    return {
                        'prompt': prompt,
                        'original': og_img,
                        'edited': edit_img,
                        'edit_type': edit_type,
                    }
                except Exception as e:
                    is_rate_limit = '429' in str(e)
                    if attempt < max_retries - 1:
                        delay = 30 * (attempt + 1) if is_rate_limit else 2 ** (attempt + 1)
                        if is_rate_limit:
                            print(f"[local_dataset] 429 rate limited idx={index}, backing off {delay}s")
                        time.sleep(delay)
                    else:
                        print(f"[local_dataset] Failed idx={index} after {max_retries} attempts: {e}")
                        return -1

        return {
            'prompt': prompt,
            'original_path_or_url': str(local_path),
            'edited_url': edit_url,
            'edit_type': edit_type,
        }

    def __len__(self):
        return len(self.data)
