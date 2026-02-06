import re
from collections.abc import Iterator
from typing import Any, Literal, get_args, get_origin

from openai.types import AllModels
from pydantic_ai.models.openai import (
    OpenAIChatModel,
    OpenAIChatModelSettings,
    OpenAIResponsesModel,
    OpenAIResponsesModelSettings,
)
from typing_extensions import TypeAliasType


def _get_literal_vals(target: Any) -> frozenset[str]:
    def _resolve(value: Any) -> Iterator[str]:
        if isinstance(value, TypeAliasType):
            yield from _resolve(value.__value__)
            return

        origin = get_origin(value)

        if origin is Literal:
            for item in get_args(value):
                if isinstance(item, str):
                    yield item
            return

        # Union[...] (incl. `typing.Union` and PEP604 unions) -> walk args.
        if origin is not None:
            for item in get_args(value):
                yield from _resolve(item)
            return

        if isinstance(value, tuple):
            for item in value:
                yield from _resolve(item)
            return

        if isinstance(value, str):
            yield value

    return frozenset(_resolve(target))


def get_variables() -> dict[str, object]:  # noqa: D103
    variables: dict[str, object] = {
        'OpenAIChatModel': OpenAIChatModel,
        'OpenAIChatModelSettings': OpenAIChatModelSettings,
        'OpenAIResponsesModel': OpenAIResponsesModel,
        'OpenAIResponsesModelSettings': OpenAIResponsesModelSettings,
    }
    for name in _get_literal_vals(AllModels):
        if isinstance(name, str):
            sanitized = re.sub(r'[^0-9A-Za-z_]+', '_', name).upper()
            sanitized = re.sub(r'_+', '_', sanitized).strip('_')
            variables[f'OPENAI_MODEL_{sanitized}'] = name

    return variables
