"""
Batch API utilities for Gemini batch processing.

Handles File API uploads, JSONL construction, batch submission,
polling, result download, and cleanup.
"""
import json, os, time, tempfile
from typing import Any, Callable
from google import genai
from google.genai import types
from system_prompt import build_system_prompt as shared_build_system_prompt


# JSON schema for the compositing recipe (must match image_compositor.py)
COMPOSITE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "base": {
            "type": "STRING",
            "description": "The base image to use as the canvas. Must be 'original' or 'edited'.",
            "enum": ["original", "edited"]
        },
        "subtract": {"type": "ARRAY", "items": {"type": "STRING"}},
        "union":    {"type": "ARRAY", "items": {"type": "STRING"}}
    },
    "required": ["base", "subtract", "union"]
}


def build_system_prompt(prompt):
    """Build the compositing system prompt for a given edit instruction.

    NOTE: Keep in sync with ImageCompositor.system_prompt in image_compositor.py.
    """
    return shared_build_system_prompt(prompt)


class BatchManager:
    def __init__(self, api_key=None, model="gemini-2.5-flash", max_retries: int = 5, retry_backoff: float = 2.0):
        self.client = genai.Client(api_key=api_key or os.environ['GOOGLE'])
        self.model = model
        self.max_retries = max_retries
        self.retry_backoff = retry_backoff

    def upload_image(self, image_bytes, mime_type="image/jpeg", display_name=None):
        """Upload image bytes to Google File API. Returns file object."""
        suffix = ".jpg" if "jpeg" in mime_type else ".png"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(image_bytes)
            tmp_path = f.name
        try:
            def _do_upload():
                return self.client.files.upload(
                    file=tmp_path,
                    config=types.UploadFileConfig(
                        display_name=display_name or "batch-img",
                        mime_type=mime_type,
                    )
                )

            return self._with_retries(_do_upload, "file upload")
        finally:
            os.unlink(tmp_path)

    def build_entry(self, key, original_file, edited_file, system_prompt_text):
        """Build a single JSONL entry for the batch request."""
        return {
            "key": key,
            "request": {
                "contents": [{
                    "parts": [
                        {"text": "Original image:"},
                        {"file_data": {"mime_type": "image/jpeg", "file_uri": original_file.uri}},
                        {"text": "Edited image:"},
                        {"file_data": {"mime_type": "image/png", "file_uri": edited_file.uri}},
                        {"text": system_prompt_text}
                    ]
                }],
                "generation_config": {
                    "response_mime_type": "application/json",
                    "response_schema": COMPOSITE_SCHEMA
                }
            }
        }

    def write_jsonl(self, entries, output_path):
        """Write batch entries to a JSONL file."""
        with open(output_path, 'w') as f:
            for entry in entries:
                f.write(json.dumps(entry) + '\n')
        return output_path

    def submit(self, jsonl_path, display_name="picobanana-batch"):
        """Upload JSONL and submit batch job. Returns job object."""
        def _upload_jsonl():
            return self.client.files.upload(
                file=jsonl_path,
                config=types.UploadFileConfig(
                    display_name=display_name,
                    mime_type='jsonl',
                )
            )

        uploaded = self._with_retries(_upload_jsonl, "JSONL upload")

        def _create_batch():
            return self.client.batches.create(
                model=self.model,
                src=uploaded.name,
                config={'display_name': display_name}
            )

        job = self._with_retries(_create_batch, "batch submission")
        print(f"Batch job submitted: {job.name}")
        return job

    def get_job(self, job_name):
        """Fetch current job status without blocking."""
        return self.client.batches.get(name=job_name)

    def poll(self, job_name, interval=30):
        """Poll batch job until completion. Returns final job object."""
        terminal = {'JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED',
                     'JOB_STATE_CANCELLED', 'JOB_STATE_EXPIRED'}
        while True:
            job = self.client.batches.get(name=job_name)
            state = job.state.name
            if state in terminal:
                print(f"Batch job finished: {state}")
                return job
            print(f"Batch status: {state}. Waiting {interval}s...")
            time.sleep(interval)

    def download_results(self, job):
        """Download and parse batch results into {key: composite_json} dict."""
        if job.state.name != 'JOB_STATE_SUCCEEDED':
            raise RuntimeError(f"Batch job did not succeed: {job.state.name}")

        def _download():
            return self.client.files.download(file=job.dest.file_name)

        content = self._with_retries(_download, f"result download ({job.name})")
        content_str = content.decode('utf-8')

        results = {}
        for line in content_str.splitlines():
            if not line.strip():
                continue
            entry = json.loads(line)
            key = entry['key']
            try:
                text = entry['response']['candidates'][0]['content']['parts'][0]['text']
                results[key] = json.loads(text)
            except (KeyError, IndexError, json.JSONDecodeError) as e:
                print(f"Failed to parse result for {key}: {e}")
                results[key] = None
        return results

    def delete_files(self, file_names):
        """Delete uploaded files from File API."""
        for name in file_names:
            try:
                def _delete():
                    return self.client.files.delete(name=name)
                self._with_retries(_delete, f"file delete {name}", swallow=True)
            except Exception as e:
                print(f"Failed to delete {name}: {e}")

    def _with_retries(self, fn: Callable[[], Any], operation: str, swallow: bool = False):
        """Retry a Google API call with exponential backoff."""
        attempt = 0
        while True:
            try:
                return fn()
            except Exception as exc:
                attempt += 1
                if attempt >= self.max_retries:
                    if swallow:
                        print(f"{operation} failed after {self.max_retries} attempts: {exc}")
                        return None
                    raise
                sleep_for = self.retry_backoff * (2 ** (attempt - 1))
                print(f"{operation} failed (attempt {attempt}/{self.max_retries}): {exc}. Retrying in {sleep_for:.1f}s...")
                time.sleep(sleep_for)
