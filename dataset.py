import PIL, json, requests, io, asyncio, aiohttp

class PicobananaDataset:
    def __init__(self, start_index=0, n=200, return_img=True):
        # Just fetch the JSON structure first (fast)
        print("Fetching JSONL index...")
        self.n = n
        self.return_img = return_img
        self.start_index = start_index
        r = requests.get('https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb/jsonl/sft.jsonl')
        self.raw_data = [json.loads(line) for line in r.text.splitlines() if line.strip()][self.start_index:self.start_index + self.n]
        self.edit_types = set()
        self.data = []
        self.fails = 0

    async def prepare_data(self):
        print(f"Validating {len(self.raw_data)} items asynchronously...")
        # Await the filter directly - no asyncio.run() needed
        self.data = await self.filter_valid_urls(self.raw_data[self.start_index:self.start_index+self.n])
        print(f"Finished. Retained {len(self.data)} valid items, {self.fails} fails")

    # (Keep your existing filter_valid_urls and check_item methods the same)
    async def filter_valid_urls(self, raw_data):
        valid_items = []
        sem = asyncio.Semaphore(50)
        async with aiohttp.ClientSession() as session:
            tasks = [self.check_item(sem, session, item, i) for i, item in enumerate(raw_data)]
            from tqdm.asyncio import tqdm
            for result in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
                item = await result
                if item:
                    valid_items.append(item)
        return valid_items

    async def check_item(self, sem, session, item, index):
        async with sem:
            try:
                url_orig = item['open_image_input_url']
                url_edit = "https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb/" + item['output_image']
                async with session.get(url_orig, timeout=10) as resp_orig:
                    if resp_orig.status != 200:
                        self.fails += 1
                        return None
                async with session.get(url_edit, timeout=10) as resp_edit:
                    if resp_edit.status != 200:
                        self.fails += 1
                        return None
                self.edit_types.add(item['edit_type'])
                return item

            except Exception as e:
                return None

    def __getitem__(self, index):
        prompt, edit_type, summarized_text = self.data[index]['text'], self.data[index]['edit_type'], self.data[index]['summarized_text']
        original_image = PIL.Image.open(io.BytesIO(requests.get(self.data[index]['open_image_input_url']).content))
        edited_image = PIL.Image.open(io.BytesIO(requests.get("https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb/"+self.data[index]['output_image']).content))
        return {'prompt':prompt,
                'original':original_image if self.return_img else self.data[index]['open_image_input_url'],
                'edited':edited_image if self.return_img else "https://ml-site.cdn-apple.com/datasets/pico-banana-300k/nb/"+self.data[index]['output_image'],
                'edit_type':edit_type}

    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    dataset = PicobananaDataset()
    asyncio.run(dataset.prepare_data())
    plt.imshow(dataset[10]['edited'])
    plt.show()