import requests
from PIL import Image
from io import BytesIO
from config import OMNIPARSER_API  # Assuming the API URL is stored in config.py


def compress_image(image_path: str, max_size=(800, 800), quality=80) -> BytesIO:
    """
    Resize and compress the image to reduce file size.
    """
    with Image.open(image_path) as img:
        img.thumbnail(max_size)

        # Create an in-memory byte stream
        img_bytes = BytesIO()

        # Save the image as JPEG to compress it further (quality=80 for good balance)
        img.save(img_bytes, format="JPEG", quality=quality)
        img_bytes.seek(0)  # Go to the beginning of the BytesIO stream
        return img_bytes


def parse_image_with_omnparser(image_path: str) -> dict:
    """
    Parse the image using OmniParser API synchronously.
    """
    try:
        # Compress the image
        compressed_image = compress_image(image_path)

        # Prepare form data
        form = {
            "prompt": "",
            "box_threshold": "0.05",
            "iou_threshold": "0.1",
            "use_paddleocr": "true",
            "imgsz": "640"
        }

        # Make a synchronous POST request to the OmniParser API
        files = {
            "image": ("screenshot.jpg", compressed_image, "image/jpeg")
        }

        # Send the request to OmniParser API
        response = requests.post(OMNIPARSER_API, data=form, files=files)

        # Check the response status
        if response.status_code != 200:
            print(f"❌ Failed to parse image {image_path}. HTTP Status: {response.status_code}")
            return None

        # Return parsed result from OmniParser
        print(f"✅ OmniParser response received for {image_path}")
        return response.json()

    except Exception as e:
        print(f"❌ Error while processing {image_path}: {e}")
        return None


def parse_image_with_retries(image_path: str, max_retries=5, retry_delay=10) -> dict:
    """
    Retry failed requests up to `max_retries` times with `retry_delay` delay between each attempt.
    """
    for attempt in range(1, max_retries + 1):
        print(f"🔁 Attempt {attempt} for {image_path}")
        result = parse_image_with_omnparser(image_path)

        if result:
            return result

        print(f"⏳ Retrying in {retry_delay} seconds...")
        time.sleep(retry_delay)

    print(f"⚠️ Failed to process {image_path} after {max_retries} attempts")
    return None
