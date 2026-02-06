import asyncio
from datetime import datetime

from task._models.custom_content import Attachment
from task._utils.constants import API_KEY, DIAL_URL, DIAL_CHAT_COMPLETIONS_ENDPOINT
from task._utils.bucket_client import DialBucketClient
from task._utils.model_client import DialModelClient
from task._models.message import Message
from task._models.role import Role

class Size:
    """
    The size of the generated image.
    """
    square: str = '1024x1024'
    height_rectangle: str = '1024x1792'
    width_rectangle: str = '1792x1024'


class Style:
    """
    The style of the generated image. Must be one of vivid or natural.
     - Vivid causes the model to lean towards generating hyper-real and dramatic images.
     - Natural causes the model to produce more natural, less hyper-real looking images.
    """
    natural: str = "natural"
    vivid: str = "vivid"


class Quality:
    """
    The quality of the image that will be generated.
     - ‘hd’ creates images with finer details and greater consistency across the image.
    """
    standard: str = "standard"
    hd: str = "hd"

async def _save_images(attachments: list[Attachment]):
    # 1. Create DIAL bucket client
    async with DialBucketClient(api_key=API_KEY, base_url=DIAL_URL) as bucket_client:
        # 2. Iterate through Images from attachments, download them and then save here
        for index, attachment in enumerate(attachments):
            if not attachment.url:
                continue

            data = await bucket_client.get_file(attachment.url)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"generated_image_{timestamp}_{index}.png"
            with open(filename, "wb") as f:
                f.write(data)

            # 3. Print confirmation that image has been saved locally
            print(f"Saved generated image to {filename}")


def start() -> None:
    # 1. Create DialModelClient using a DALL‑E 3 compatible deployment
    client = DialModelClient(
        endpoint=DIAL_CHAT_COMPLETIONS_ENDPOINT,
        deployment_name="dall-e-3",
        api_key=API_KEY,
    )

    # 2. Generate image for "Sunny day on Bali"
    prompt = "Generate a vivid illustration of a sunny day on Bali, with beaches, palm trees, and clear blue water."
    message = Message(
        role=Role.USER,
        content=prompt,
    )

    # 4. Configure the picture for output via `custom_fields` parameter.
    custom_fields = {
        "size": Size.square,
        "quality": Quality.standard,
        "style": Style.vivid,
    }

    # Google image generation model example: 'imagegeneration@005'
    # To test with it, change deployment_name above to "imagegeneration@005".

    response = client.get_completion(
        messages=[message],
        custom_fields=custom_fields,
    )

    # 3. Get attachments from response and save generated images (use method `_save_images`)
    attachments = response.custom_content.attachments if response.custom_content else []
    if attachments:
        asyncio.run(_save_images(attachments))
    else:
        print("No attachments with generated images were returned by the model.")


start()
