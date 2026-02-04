import PIL.Image, json, requests, io
import matplotlib.pyplot as plt
import asyncio
import aiohttp
from pathlib import Path

class PicobananaDataset:
    def __init__(self, start_index=0, n=200, return_img=True, local_dir="openimage_source_images"):
        print("Fetching JSONL index...")
        self.n = n
        self.return_img = return_img
        self.start_index = start_index
        self.local_dir = Path(local_dir)
        
        # Load JSONL
        r = requests.get('https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb/jsonl/sft.jsonl')
        all_lines = [json.loads(line) for line in r.text.splitlines() if line.strip()]
        self.raw_data = all_lines[self.start_index : self.start_index + self.n]
        
        self.edit_types = set()
        self.data = []
        self.fails = 0

    async def prepare_data(self):
        print(f"Checking {len(self.raw_data)} items...")
        # We don't need to check URLs for local files, so we validate Apple's CDN status
        self.data = await self.filter_valid_items(self.raw_data)
        print(f"Finished. Retained {len(self.data)} valid items, {self.fails} fails")

    async def filter_valid_items(self, raw_data):
        valid_items = []
        sem = asyncio.Semaphore(20) # Conservative to avoid CDN bans
        async with aiohttp.ClientSession() as session:
            tasks = [self.check_item(sem, session, item) for item in raw_data]
            from tqdm.asyncio import tqdm
            for result in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
                item = await result
                if item:
                    valid_items.append(item)
        return valid_items

    async def check_item(self, sem, session, item):
        async with sem:
            try:
                # 1. Check Local vs URL for Original
                img_id = item['open_image_input_url'].split('/')[-1]
                local_path = self.local_dir / img_id
                
                # If not local, check if URL is still alive
                if not local_path.exists():
                    async with session.head(item['open_image_input_url'], timeout=5) as resp:
                        if resp.status != 200:
                            self.fails += 1
                            return None

                # 2. Check Apple's Edited Image URL
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
        
        # Determine source of Original Image
        img_id = item['open_image_input_url'].split('/')[-1]
        local_path = self.local_dir / img_id
        edit_url = "https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb/" + item['output_image']

        if self.return_img:
            try:
                # Load Original
                if local_path.exists():
                    og_img = PIL.Image.open(local_path).convert("RGB")
                else:
                    resp = requests.get(item['open_image_input_url'], timeout=10)
                    og_img = PIL.Image.open(io.BytesIO(resp.content)).convert("RGB")
                
                # Load Edited (Always from Web in this setup)
                resp_edit = requests.get(edit_url, timeout=10)
                edit_img = PIL.Image.open(io.BytesIO(resp_edit.content)).convert("RGB")
                
                return {
                    'prompt': prompt,
                    'original': og_img,
                    'edited': edit_img,
                    'edit_type': edit_type
                }
            except:
                return -1
        
        return {
            'prompt': prompt,
            'original_path_or_url': str(local_path) if local_path.exists() else item['open_image_input_url'],
            'edited_url': edit_url,
            'edit_type': edit_type
        }

    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    import asyncio
    dataset = PicobananaDataset(n=500) # Bumped n since local checking is fast
    asyncio.run(dataset.prepare_data())
    import requests
    import matplotlib.pyplot as plt

    datum = dataset[0]
    print(datum['prompt'])
    plt.imshow(datum['original'])
    plt.show()