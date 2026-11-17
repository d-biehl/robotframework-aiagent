import re

from pydantic_ai.models.openrouter import (
    KnownOpenRouterProviders,
    OpenRouterModel,
    OpenRouterModelSettings,
    OpenRouterProviderConfig,
    OpenRouterReasoning,
    OpenRouterStreamedResponse,
    OpenRouterUsageConfig,
)
from pydantic_ai.providers.openrouter import OpenRouterProvider

from AIAgent._literals import literal_values


def get_variables() -> dict[str, object]:  # noqa: D103
    variables: dict[str, object] = {
        'OpenRouterModel': OpenRouterModel,
        'OpenRouterModelSettings': OpenRouterModelSettings,
        'OpenRouterProvider': OpenRouterProvider,
        'OpenRouterProviderConfig': OpenRouterProviderConfig,
        'OpenRouterReasoning': OpenRouterReasoning,
        'OpenRouterStreamedResponse': OpenRouterStreamedResponse,
        'OpenRouterUsageConfig': OpenRouterUsageConfig,
    }
    for name in literal_values(KnownOpenRouterProviders):
        sanitized = re.sub(r'[^0-9A-Za-z_]+', '_', name).upper()
        sanitized = re.sub(r'_+', '_', sanitized).strip('_')
        variables[f'OPENROUTER_PROVIDER_{sanitized}'] = name

    return variables
