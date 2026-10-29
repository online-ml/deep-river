from collections.abc import Hashable, Iterable, Iterator, MutableSet
from typing import Generic, TypeVar

T = TypeVar("T", bound=Hashable)


class OrderedSet(MutableSet[T], Generic[T]):
    def __init__(self, values: Iterable[T] = ()):
        self._indices: dict[T, int] = {}
        for value in values:
            self.add(value)

    def __contains__(self, value: object) -> bool:
        return value in self._indices

    def __iter__(self) -> Iterator[T]:
        return iter(self._indices)

    def __len__(self) -> int:
        return len(self._indices)

    def add(self, value: T) -> None:
        if value not in self._indices:
            self._indices[value] = len(self._indices)

    def discard(self, value: T) -> None:
        if value in self._indices:
            del self._indices[value]
            self._indices = {item: index for index, item in enumerate(self._indices)}

    def clear(self) -> None:
        self._indices.clear()

    def update(self, values: Iterable[T]) -> None:
        for value in sorted(set(values) - self._indices.keys(), key=repr):
            self.add(value)

    def index(self, value: T) -> int:
        try:
            return self._indices[value]
        except KeyError:
            raise ValueError(value) from None
