# omniparser_client.py

import aiohttp
import asyncio
import time
from aiohttp import FormData, ClientTimeout
from PIL import Image
from io import BytesIO
from config import OMNIPARSER_API


MAX_RETRIES = 5  # Max retries for failed requests
RETRY_DELAY = 10  # Delay in seconds between retries


async def compress_image(image_path: str, max_size=(800, 800), quality=80) -> BytesIO:
    """
    Resizes and compresses the image to reduce file size.
    """
    with Image.open(image_path) as img:
        img.thumbnail(max_size)

        # Create an in-memory byte stream
        img_bytes = BytesIO()

        # Save the image as JPEG to compress it further (quality=80 for good balance)
        img.save(img_bytes, format="JPEG", quality=quality)
        img_bytes.seek(0)  # Go to the beginning of the BytesIO stream
        return img_bytes


async def parse_image_with_omnparser(image_path: str) -> dict:
    """
    Parse the image using OmniParser API.
    """
    compressed_image = await compress_image(image_path)

    form = FormData()
    form.add_field("prompt", "")
    form.add_field("box_threshold", "0.05")
    form.add_field("iou_threshold", "0.1")
    form.add_field("use_paddleocr", "true")
    form.add_field("imgsz", "640")

    # Add compressed image to form
    form.add_field(
        name="image",
        value=compressed_image,
        filename="screenshot.jpg",  # Now a .jpg file (compressed)
        content_type="image/jpeg"
    )

    # Increase timeout to 60 seconds
    timeout = ClientTimeout(total=600)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.post(OMNIPARSER_API, data=form) as response:
                if response.status != 200:
                    print(f"❌ HTTP {response.status} for {image_path}")
                    return None
                print(f"✅ Parsed output received from OmniParser for {image_path}")
                return await response.json()
        except asyncio.TimeoutError:
            print(f"⏰ Timeout while processing {image_path}")
            return None


async def parse_image_with_retries(image_path: str) -> dict:
    """
    Retry failed requests up to `MAX_RETRIES` times with `RETRY_DELAY` delay between each.
    """
    for attempt in range(MAX_RETRIES):
        result = await parse_image_with_omnparser(image_path)
        if result:
            return result

        # Log retry information
        print(f"❌ Retrying {image_path}... (Attempt {attempt + 1}/{MAX_RETRIES})")
        await asyncio.sleep(RETRY_DELAY)

    print(f"⚠️ Failed to process {image_path} after {MAX_RETRIES} attempts")
    return None
