import re

from openai.types import AllModels
from pydantic_ai.models.openai import (
    OpenAIChatModel,
    OpenAIChatModelSettings,
    OpenAIResponsesModel,
    OpenAIResponsesModelSettings,
)

from AIAgent._literals import literal_values


def get_variables() -> dict[str, object]:  # noqa: D103
    variables: dict[str, object] = {
        'OpenAIChatModel': OpenAIChatModel,
        'OpenAIChatModelSettings': OpenAIChatModelSettings,
        'OpenAIResponsesModel': OpenAIResponsesModel,
        'OpenAIResponsesModelSettings': OpenAIResponsesModelSettings,
    }
    for name in literal_values(AllModels):
        sanitized = re.sub(r'[^0-9A-Za-z_]+', '_', name).upper()
        sanitized = re.sub(r'_+', '_', sanitized).strip('_')
        variables[f'OPENAI_MODEL_{sanitized}'] = name

    return variables
