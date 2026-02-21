from google import genai
import os, time
from dotenv import load_dotenv

load_dotenv()
client = genai.Client(api_key=os.environ['GOOGLE'])

deleted = 0
errors = 0

# Iterate page by page instead of loading all at once
while True:
    try:
        batch = []
        for f in client.files.list():
            batch.append(f)
            if len(batch) >= 100:
                break

        if not batch:
            print(f"No more files. Deleted {deleted} total, {errors} errors.")
            break

        print(f"Deleting batch of {len(batch)} files...")
        for f in batch:
            try:
                client.files.delete(name=f.name)
                deleted += 1
            except Exception as e:
                errors += 1
                if '429' in str(e):
                    print(f"Rate limited, sleeping 10s...")
                    time.sleep(10)
                    try:
                        client.files.delete(name=f.name)
                        deleted += 1
                        errors -= 1
                    except:
                        pass

        if deleted % 500 == 0:
            print(f"  Progress: {deleted} deleted, {errors} errors")

    except Exception as e:
        print(f"List error: {e}, sleeping 30s and retrying...")
        time.sleep(30)

print(f"Done. Deleted {deleted} files, {errors} errors.")
