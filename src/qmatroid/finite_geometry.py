"""The finite-geometry data used in the classification."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cache
import gzip
from importlib import resources
import io
import json
from types import MappingProxyType
from typing import Iterable, Mapping

from sage.all import matrix


SUBSPACE_ORDER_VERSION = "v1"


def subspaces_iterator(V, dimension=None):
    """Iterate over the subspaces of ``V``, grouped by dimension."""
    dimensions = range(V.dimension() + 1) if dimension is None else (dimension,)
    for k in dimensions:
        yield from V.subspaces(k)


class ReversedLexicographicOrderOfSubspaces:
    """The order on subspaces used in the paper.

    The elements of the field are ordered as

    ``0 < 1 < g < g^2 < ... < g^(q - 2)``,

    where ``g`` is the multiplicative generator chosen by SageMath. A
    subspace is represented by its reversed canonical matrix, and these
    matrices are compared in reversed lexicographic order.
    """

    def __init__(self, V):
        self.V = V
        F = V.base_field()
        g = F.multiplicative_generator()
        self.field_index = {
            a: i
            for i, a in enumerate(
                (F.zero(), F.one(), *(g**j for j in range(1, F.order() - 1)))
            )
        }

    def canonical_reversed_representation(self, X):
        """Return the reversed canonical matrix representing ``X``."""
        if X.dimension() == 0:
            return matrix(self.V.base_field(), 0, self.V.dimension())
        A = X.basis_matrix()
        return A[::-1, ::-1].echelon_form()[::-1, ::-1]

    def _vector_key(self, x):
        return tuple(self.field_index[a] for a in reversed(x))

    def _matrix_key(self, A):
        return tuple(self._vector_key(x) for x in reversed(A.rows()))

    def key(self, X):
        """Return a sorting key for ``X``."""
        return self._matrix_key(self.canonical_reversed_representation(X))

    def sorted(self, dimension=None):
        """Return the subspaces in the prescribed order."""
        return sorted(subspaces_iterator(self.V, dimension), key=self.key)


def q_binomial(q, n, k):
    """Return the Gaussian binomial coefficient."""
    if not 0 <= k <= n:
        return 0
    numerator = 1
    denominator = 1
    for i in range(k):
        numerator *= q ** (n - i) - 1
        denominator *= q ** (i + 1) - 1
    return numerator // denominator


def number_of_subspaces_in_ambient_space(q, n):
    """Return the number of subspaces of ``GF(q)^n``."""
    return sum(q_binomial(q, n, k) for k in range(n + 1))


@dataclass(frozen=True, slots=True)
class SubspaceTables:
    """Precomputed subspaces and their elementary operations.

    A position in each table is the ID of a subspace in
    :class:`ReversedLexicographicOrderOfSubspaces`.
    """

    q: int
    max_dimension: int
    order_version: str
    subspace_dimensions: tuple[int, ...]
    contained_subspace_ids: tuple[tuple[int, ...], ...]
    contained_point_ids: tuple[tuple[int, ...], ...]
    span_ids_by_point: tuple[tuple[int, ...], ...]
    reversed_canonical_basis_point_ids: tuple[tuple[int, ...], ...]
    _point_ids: tuple[int, ...] = field(init=False, repr=False, compare=False)
    _point_column: Mapping[int, int] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _ids_by_dimension: tuple[tuple[int, ...], ...] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _standard_space_ids: tuple[int, ...] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _point_sets: tuple[frozenset[int], ...] = field(
        init=False,
        repr=False,
        compare=False,
    )
    _id_by_point_set: Mapping[frozenset[int], int] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self):
        point_ids = tuple(
            X for X, k in enumerate(self.subspace_dimensions) if k == 1
        )
        point_sets = tuple(map(frozenset, self.contained_point_ids))
        object.__setattr__(self, "_point_ids", point_ids)
        object.__setattr__(
            self,
            "_point_column",
            MappingProxyType({e: i for i, e in enumerate(point_ids)}),
        )
        object.__setattr__(
            self,
            "_ids_by_dimension",
            tuple(
                tuple(
                    X
                    for X, dimension in enumerate(self.subspace_dimensions)
                    if dimension == k
                )
                for k in range(self.max_dimension + 1)
            ),
        )
        object.__setattr__(
            self,
            "_standard_space_ids",
            tuple(
                next(
                    X
                    for X in range(
                        number_of_subspaces_in_ambient_space(self.q, n)
                    )
                    if self.subspace_dimensions[X] == n
                )
                for n in range(self.max_dimension + 1)
            ),
        )
        object.__setattr__(self, "_point_sets", point_sets)
        object.__setattr__(
            self,
            "_id_by_point_set",
            MappingProxyType({points: X for X, points in enumerate(point_sets)}),
        )

    @property
    def number_of_subspaces(self):
        return len(self.subspace_dimensions)

    @property
    def point_ids(self):
        return self._point_ids

    def subspace_ids(self, dimension):
        return self._ids_by_dimension[dimension]

    def standard_space_id(self, dimension):
        return self._standard_space_ids[dimension]

    def subspace_id_from_point_ids(self, point_ids: Iterable[int]):
        return self._id_by_point_set[frozenset(point_ids)]

    def span_id(self, X, e):
        return self.span_ids_by_point[X][self._point_column[e]]

    def sum_id(self, X, Y):
        for e in self.reversed_canonical_basis_point_ids[Y]:
            X = self.span_id(X, e)
        return X

    def intersection_id(self, X, Y):
        return self._id_by_point_set[self._point_sets[X] & self._point_sets[Y]]


def _rows(values):
    return tuple(tuple(row) for row in values)


@cache
def _precomputed_tables(q):
    resource = resources.files("qmatroid")
    for part in ("data", SUBSPACE_ORDER_VERSION, f"q{q}", "tables.json.gz"):
        resource = resource.joinpath(part)
    with resource.open("rb") as source:
        with gzip.GzipFile(mode="rb", fileobj=source) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8") as text_file:
                data = json.load(text_file)

    return SubspaceTables(
        q=data["q"],
        max_dimension=data["max_dimension"],
        order_version=data["order_version"],
        subspace_dimensions=tuple(data["subspace_dimensions"]),
        contained_subspace_ids=_rows(data["contained_subspace_ids"]),
        contained_point_ids=_rows(data["contained_point_ids"]),
        span_ids_by_point=_rows(data["span_ids_by_point"]),
        reversed_canonical_basis_point_ids=_rows(
            data["reverse_canonical_basis_point_ids"]
        ),
    )


def get_subspace_tables(q, max_dimension):
    """Return the bundled tables covering ``GF(q)^max_dimension``."""
    supported_dimensions = {2: 5, 3: 4}
    if (
        q not in supported_dimensions
        or not 0 <= max_dimension <= supported_dimensions[q]
    ):
        raise ValueError("precomputed tables cover q=2 through n=5 and q=3 through n=4")
    return _precomputed_tables(q)


__all__ = [
    "ReversedLexicographicOrderOfSubspaces",
    "SUBSPACE_ORDER_VERSION",
    "SubspaceTables",
    "get_subspace_tables",
    "number_of_subspaces_in_ambient_space",
    "q_binomial",
    "subspaces_iterator",
]
