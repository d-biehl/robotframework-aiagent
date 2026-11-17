"""Read the string values of `Literal` types, e.g. the model names an SDK knows."""

from collections.abc import Iterator
from typing import Any, Literal, get_args, get_origin

from typing_extensions import TypeAliasType


def literal_values(target: Any) -> frozenset[str]:
    """Collect the string values of a `Literal`, following type aliases, unions and tuples."""

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
