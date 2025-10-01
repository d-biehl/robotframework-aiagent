import base64
from pathlib import Path
from typing import Literal

from openai import OpenAI
from pydantic_ai.toolsets import FunctionToolset
from robot.api import logger
from robot.libraries.BuiltIn import BuiltIn

client = OpenAI()


def gen_image(
    prompt: str,
    size: Literal['auto', '1024x1024', '1536x1024', '1024x1536', '256x256', '512x512', '1792x1024', '1024x1792']
    | None = None,
    file_name_base: str = 'output',
    n: int | None = None,
) -> list[str]:
    """Generate one or more images with the OpenAI `gpt-image-1` model and save them as PNG files.

    Args:
        prompt: Natural language description of the desired image.
        size: Optional image size. If ``NOT_GIVEN`` the provider's default is used. Allowed values:
            'auto', '1024x1024', '1536x1024', '1024x1536', '256x256', '512x512', '1792x1024', '1024x1792'.
        file_name_base: Base name for the output files. Each generated image is saved as
            ``{file_name_base}_{index}.png`` in the current working directory.
        n: Optional number of images to generate. If ``None`` the provider's default is used.

    Returns:
        list[str]: A list of the PNG file paths written to disk.

    Raises:
        ValueError: If the API response contains no image data or an item lacks ``b64_json`` content.

    Example:
        >>> gen_image("Minimal blue square logo", size="512x512", file_name_base="logo")
        ['logo_0.png']
    """
    resp = client.images.generate(prompt=prompt, model='gpt-image-1', size='auto', output_format='png', n=n)

    if not resp.data:
        raise ValueError('No image data returned from OpenAI')

    builtin = BuiltIn()
    output_dir: str | None = builtin.get_variable_value('${OUTPUT_DIR}', None)
    if output_dir is not None and output_dir != '':
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        file_name_base = str(Path(output_dir) / file_name_base)

    result: list[str] = []
    for image in resp.data:
        if image.b64_json is None:
            raise ValueError('No image data returned from OpenAI')

        file_name = f'{file_name_base}_{len(result)}.png'
        result.append(file_name)

        logger.info(f'Generated image saved to {file_name}')
        logger.info(f'<img src="{Path(file_name).name}" alt="Generated image" />', html=True)

        b64 = image.b64_json
        Path(file_name).write_bytes(base64.b64decode(b64))

    return result


genimage_toolset = FunctionToolset()
genimage_toolset.add_function(gen_image)

__all__ = ['genimage_toolset']
