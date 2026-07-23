from __future__ import annotations

from typing import Any, Mapping, Sequence


QueryValues = Mapping[str, Sequence[Any]]


class ApiValidationError(ValueError):
    """A request query value could not be normalized safely."""


def query_value(
    query: QueryValues,
    name: str,
    default: str = "",
) -> str:
    values = query.get(name)
    if not values:
        return str(default)
    return str(values[0] if values[0] is not None else default)


def query_int(
    query: QueryValues,
    name: str,
    *,
    default: int,
    minimum: int | None = None,
    maximum: int | None = None,
    clamp: bool = False,
) -> int:
    raw = query_value(query, name, str(default)).strip()
    try:
        value = int(raw or default)
    except (TypeError, ValueError) as exc:
        raise ApiValidationError(f"{name} must be an integer") from exc
    if minimum is not None and value < minimum:
        if clamp:
            value = minimum
        else:
            raise ApiValidationError(f"{name} must be at least {minimum}")
    if maximum is not None and value > maximum:
        if clamp:
            value = maximum
        else:
            raise ApiValidationError(f"{name} must be at most {maximum}")
    return value


def pagination_params(
    query: QueryValues,
    *,
    default_page_size: int,
    maximum_page_size: int = 200,
) -> tuple[int, int]:
    """Normalize the existing clamp-to-range paging contract."""

    page = query_int(
        query,
        "page",
        default=1,
        minimum=1,
        clamp=True,
    )
    page_size = query_int(
        query,
        "page_size",
        default=default_page_size,
        minimum=1,
        maximum=maximum_page_size,
        clamp=True,
    )
    return page, page_size
