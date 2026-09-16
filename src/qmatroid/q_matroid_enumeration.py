"""Rank-table q-matroids and their one-dimensional extensions."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cache
from itertools import combinations, islice
import multiprocessing
from pathlib import Path
from typing import Iterable, Iterator, Mapping

from sage.all import (
    GF,
    GL,
    MatrixGroup,
    PermutationGroup,
    VectorSpace,
    identity_matrix,
    libgap,
    matrix,
)

from qmatroid.finite_geometry import (
    ReversedLexicographicOrderOfSubspaces,
    SubspaceTables,
    get_subspace_tables,
    number_of_subspaces_in_ambient_space,
)


ModularCut = frozenset[int]
ModularCutSelector = dict[int, ModularCut]


_CANONICAL_WORKER_TABLES: SubspaceTables | None = None


@cache
def _orthogonal_complement_ids(q: int, n: int) -> tuple[int, ...]:
    """Return orthogonal-complement IDs in the fixed subspace order."""
    V = VectorSpace(GF(q), n)
    subspaces = tuple(ReversedLexicographicOrderOfSubspaces(V).sorted())
    position = {X: i for i, X in enumerate(subspaces)}
    return tuple(
        position[X.basis_matrix().right_kernel()]
        for X in subspaces
    )


@dataclass(frozen=True, slots=True)
class RankTableQMatroid:
    """A q-matroid whose rank values follow the fixed subspace-ID order."""

    q: int
    n: int
    rank_table: tuple[int, ...]

    @classmethod
    def from_basis_ids(
        cls,
        q: int,
        n: int,
        basis_ids: Iterable[int],
        tables: SubspaceTables,
    ) -> RankTableQMatroid:
        """Construct the rank table defined by a family of bases."""
        basis_ids = tuple(basis_ids)
        table_length = number_of_subspaces_in_ambient_space(q, n)
        rank_table = tuple(
            max(
                tables.subspace_dimensions[
                    tables.intersection_id(subspace_id, basis_id)
                ]
                for basis_id in basis_ids
            )
            for subspace_id in range(table_length)
        )
        return cls(q, n, rank_table)

    def __post_init__(self) -> None:
        object.__setattr__(self, "rank_table", tuple(self.rank_table))

    def rank(self, subspace_id: int) -> int:
        """Return the rank at ``subspace_id`` by one tuple lookup."""
        return self.rank_table[subspace_id]

    def ground_space_id(self, tables: SubspaceTables) -> int:
        """Return the ID of the standard ground space ``GF(q)^n``."""
        return tables.standard_space_id(self.n)

    def full_rank(self, tables: SubspaceTables) -> int:
        """Return the rank of the ground space."""
        return self.rank(self.ground_space_id(tables))

    def is_valid(self, tables: SubspaceTables) -> bool:
        """Return whether the rank table satisfies the q-matroid rank axioms."""
        subspaces = range(len(self.rank_table))
        for X in subspaces:
            if not 0 <= self.rank(X) <= tables.subspace_dimensions[X]:
                return False
        for Y in subspaces:
            for X in tables.contained_subspace_ids[Y]:
                if self.rank(X) > self.rank(Y):
                    return False
        for X, Y in combinations(subspaces, 2):
            if self.rank(X) + self.rank(Y) < (
                self.rank(tables.sum_id(X, Y))
                + self.rank(tables.intersection_id(X, Y))
            ):
                return False
        return True

    def is_independent(self, X: int, tables: SubspaceTables) -> bool:
        """Return whether a subspace is independent."""
        return self.rank(X) == tables.subspace_dimensions[X]

    def is_basis(self, X: int, tables: SubspaceTables) -> bool:
        """Return whether a subspace is a basis of the q-matroid."""
        return (
            tables.subspace_dimensions[X] == self.full_rank(tables)
            and self.is_independent(X, tables)
        )

    def basis_ids(self, tables: SubspaceTables) -> tuple[int, ...]:
        """Return basis IDs in the fixed subspace order."""
        k = self.full_rank(tables)
        return tuple(
            X
            for X in range(len(self.rank_table))
            if tables.subspace_dimensions[X] == k and self.rank(X) == k
        )

    def basis_encoding(self, tables: SubspaceTables) -> tuple[int, ...]:
        """Return the binary basis encoding in the fixed subspace order."""
        k = self.full_rank(tables)
        return tuple(
            int(self.rank(X) == k)
            for X in tables.subspace_ids(k)
            if X < len(self.rank_table)
        )

    def dual(self, tables: SubspaceTables) -> RankTableQMatroid:
        """Return the dual defined by the standard bilinear form."""
        k = self.full_rank(tables)
        dual_ids = _orthogonal_complement_ids(self.q, self.n)
        return RankTableQMatroid(
            self.q,
            self.n,
            tuple(
                tables.subspace_dimensions[X] - k + self.rank(dual_ids[X])
                for X in range(len(self.rank_table))
            ),
        )

    def is_canonical(self, tables: SubspaceTables) -> bool:
        """Return whether the basis encoding is minimal under the GL action."""
        encoding = self.basis_encoding(tables)
        if len(set(encoding)) <= 1:
            return True

        nonbases = tuple(i for i, value in enumerate(encoding, start=1) if not value)
        group = _gap_subspace_action_group(
            self.q,
            self.n,
            self.full_rank(tables),
        )
        return bool(libgap.IsMinimalImage(group, nonbases, libgap.OnSets))

    def canonical_representative(
        self,
        tables: SubspaceTables,
    ) -> RankTableQMatroid:
        """Return the canonical representative under the GL action."""
        encoding = self.basis_encoding(tables)
        if len(set(encoding)) <= 1:
            return self

        k = self.full_rank(tables)
        nonbases = tuple(i for i, value in enumerate(encoding, start=1) if not value)
        group = _gap_subspace_action_group(self.q, self.n, k)
        minimal_nonbases = frozenset(
            map(int, libgap.MinimalImage(group, nonbases, libgap.OnSets))
        )
        rank_subspaces = tuple(
            X for X in tables.subspace_ids(k) if X < len(self.rank_table)
        )
        bases = tuple(
            X
            for i, X in enumerate(rank_subspaces, start=1)
            if i not in minimal_nonbases
        )
        return type(self).from_basis_ids(self.q, self.n, bases, tables)

    def projective_automorphism_group(self, tables: SubspaceTables):
        """Return the projective automorphism group acting on point IDs."""
        return _projective_automorphism_group(self, tables)

    def linear_automorphism_group(self, tables: SubspaceTables):
        """Return the subgroup of ``GL(n, q)`` preserving this q-matroid."""
        return _linear_automorphism_group(self, tables)

    def is_flat(self, X: int, tables: SubspaceTables) -> bool:
        """Return whether a subspace is a flat."""
        E = self.ground_space_id(tables)
        points = frozenset(tables.contained_point_ids[X])
        return all(
            self.rank(X) < self.rank(tables.span_id(X, e))
            for e in tables.contained_point_ids[E]
            if e not in points
        )

    def flat_ids(
        self,
        tables: SubspaceTables,
        rank: int | None = None,
    ) -> tuple[int, ...]:
        """Return flat IDs in the fixed subspace order."""
        E = self.ground_space_id(tables)
        points = tables.contained_point_ids[E]
        return tuple(
            X
            for X in range(len(self.rank_table))
            if (rank is None or self.rank(X) == rank)
            and all(
                self.rank(X) < self.rank(tables.span_id(X, e))
                for e in points
                if e not in tables.contained_point_ids[X]
            )
        )

    def hyperplane_ids(self, tables: SubspaceTables) -> tuple[int, ...]:
        """Return hyperplane IDs in the fixed subspace order."""
        k = self.full_rank(tables)
        if k == 0:
            return ()
        return self.flat_ids(tables, rank=k - 1)

    def closure_id(self, X: int, tables: SubspaceTables) -> int:
        """Return the ID of the closure of a subspace."""
        return next(
            F
            for F in self.flat_ids(tables, rank=self.rank(X))
            if X in tables.contained_subspace_ids[F]
        )

    def is_modular_pair(
        self,
        F1: int,
        F2: int,
        tables: SubspaceTables,
    ) -> bool:
        """Return whether two flats form a modular pair."""
        return (
            self.rank(F1) + self.rank(F2)
            == self.rank(tables.sum_id(F1, F2))
            + self.rank(tables.intersection_id(F1, F2))
        )

    def modular_cuts(self, tables: SubspaceTables) -> Iterator[ModularCut]:
        """Yield all modular cuts in deterministic flat-ID order."""
        flats = self.flat_ids(tables)
        upper_flats = {
            F: frozenset(
                G for G in flats if F in tables.contained_subspace_ids[G]
            )
            for F in flats
        }
        for antichain in _flat_antichains(flats, tables):
            cut = frozenset(
                G for F in antichain for G in upper_flats[F]
            )
            if all(
                not self.is_modular_pair(F1, F2, tables)
                or tables.intersection_id(F1, F2) in cut
                for F1, F2 in combinations(cut, 2)
            ):
                yield cut

    def external_point_ids(self, tables: SubspaceTables) -> tuple[int, ...]:
        """Return the new points in the next standard ambient space."""
        E = self.ground_space_id(tables)
        extended_E = tables.standard_space_id(self.n + 1)
        old_points = frozenset(tables.contained_point_ids[E])
        return tuple(
            e
            for e in tables.contained_point_ids[extended_E]
            if e not in old_points
        )

    def trivial_extension(self, tables: SubspaceTables) -> RankTableQMatroid:
        """Return the trivial one-dimensional extension."""
        selector = {e: frozenset() for e in self.external_point_ids(tables)}
        return self._extension_from_selector(
            selector,
            self._extension_rank_terms(tables),
        )

    def _extension_rank_terms(
        self,
        tables: SubspaceTables,
    ) -> tuple[tuple[int, int, int], ...]:
        """Return ``(r(X ∩ E), cl(X ∩ E), e)`` for each new subspace ``X``."""
        E = self.ground_space_id(tables)
        old_points = frozenset(tables.contained_point_ids[E])
        closures = tuple(
            self.closure_id(X, tables)
            for X in range(len(self.rank_table))
        )
        rank_terms = []
        for X in range(
            len(self.rank_table),
            number_of_subspaces_in_ambient_space(self.q, self.n + 1),
        ):
            X_M = tables.intersection_id(X, E)
            e = next(
                e for e in tables.contained_point_ids[X] if e not in old_points
            )
            rank_terms.append((self.rank(X_M), closures[X_M], e))
        return tuple(rank_terms)

    def _extension_from_selector(
        self,
        selector: Mapping[int, ModularCut],
        rank_terms: tuple[tuple[int, int, int], ...],
    ) -> RankTableQMatroid:
        ranks = self.rank_table + tuple(
            rank + int(closure not in selector[e])
            for rank, closure, e in rank_terms
        )
        return RankTableQMatroid(self.q, self.n + 1, ranks)


def _flat_antichains(
    flat_ids: tuple[int, ...],
    tables: SubspaceTables,
) -> Iterator[tuple[int, ...]]:
    def visit(
        antichain: tuple[int, ...],
        candidates: tuple[int, ...],
    ) -> Iterator[tuple[int, ...]]:
        yield antichain
        for position, flat_id in enumerate(candidates):
            remaining_candidates = tuple(
                candidate_id
                for candidate_id in candidates[position + 1 :]
                if flat_id not in tables.contained_subspace_ids[candidate_id]
                and candidate_id not in tables.contained_subspace_ids[flat_id]
            )
            yield from visit(antichain + (flat_id,), remaining_candidates)

    yield from visit((), flat_ids)


def generate_modular_cut_selectors(
    M: RankTableQMatroid,
    tables: SubspaceTables,
) -> Iterator[ModularCutSelector]:
    """Yield all modular cut selectors of ``M``."""
    flat_id_set = frozenset(M.flat_ids(tables))
    modular_cuts = tuple(M.modular_cuts(tables))
    external_point_ids = M.external_point_ids(tables)
    partial_selector: ModularCutSelector = {}
    # Record the flats forced into each cut by condition (QM).
    required_flat_ids_by_point = {
        point_id: set() for point_id in external_point_ids
    }
    # Once a flat F is selected, remember its unique image F + e.
    flat_image_by_id: dict[int, int | None] = {
        flat_id: None for flat_id in flat_id_set
    }
    seen_point_ids: set[int] = set()

    def extend_selector(position: int) -> Iterator[ModularCutSelector]:
        point_id = external_point_ids[position]
        required_flat_ids = required_flat_ids_by_point[point_id]
        for modular_cut in modular_cuts:
            if not required_flat_ids <= modular_cut:
                continue

            # A selected flat must have the same image at every earlier point
            # whose cut contains it, as required by (QM).
            is_compatible = True
            for flat_id in modular_cut:
                image_id = tables.span_id(flat_id, point_id)
                known_image_id = flat_image_by_id[flat_id]
                if known_image_id is not None:
                    if image_id != known_image_id:
                        is_compatible = False
                        break
                elif any(
                    # A new image F + e cannot contain an earlier point whose
                    # cut was fixed without F.
                    affected_point_id in seen_point_ids
                    for affected_point_id in (
                        frozenset(tables.contained_point_ids[image_id])
                        - frozenset(tables.contained_point_ids[flat_id])
                    )
                ):
                    is_compatible = False
                    break
            if not is_compatible:
                continue

            # These flats acquire their image F + e for the first time.
            added_requirements: list[tuple[int, int]] = []
            newly_mapped_flat_ids = sorted(
                modular_cut - required_flat_ids
            )
            for flat_id in newly_mapped_flat_ids:
                image_id = tables.span_id(flat_id, point_id)
                flat_image_by_id[flat_id] = image_id
                affected_point_ids = (
                    frozenset(tables.contained_point_ids[image_id])
                    - frozenset(tables.contained_point_ids[flat_id])
                )
                # Condition (QM) forces F into every cut indexed by a point
                # in (F + e) \ F.
                for affected_point_id in affected_point_ids:
                    affected_requirements = required_flat_ids_by_point[
                        affected_point_id
                    ]
                    if flat_id not in affected_requirements:
                        affected_requirements.add(flat_id)
                        added_requirements.append((affected_point_id, flat_id))

            partial_selector[point_id] = modular_cut
            if position + 1 == len(external_point_ids):
                yield dict(partial_selector)
            else:
                seen_point_ids.add(point_id)
                yield from extend_selector(position + 1)
                seen_point_ids.remove(point_id)
            del partial_selector[point_id]

            # Restore the mutable DFS state before trying the next cut.
            for affected_point_id, flat_id in reversed(added_requirements):
                required_flat_ids_by_point[affected_point_id].remove(flat_id)
            for flat_id in newly_mapped_flat_ids:
                flat_image_by_id[flat_id] = None

    yield from extend_selector(0)


def generate_one_dimensional_extensions(
    M: RankTableQMatroid,
    tables: SubspaceTables,
) -> Iterator[RankTableQMatroid]:
    """Yield the one-dimensional extensions of ``M``."""
    rank_terms = M._extension_rank_terms(tables)
    for selector in generate_modular_cut_selectors(M, tables):
        yield M._extension_from_selector(selector, rank_terms)


def save_q_matroids(Ms: Iterable[RankTableQMatroid], path, mode="w") -> int:
    """Write one rank table per line and return the number written."""
    count = 0
    with Path(path).open(mode, encoding="ascii", newline="\n") as output:
        for M in Ms:
            output.write(" ".join(map(str, M.rank_table)) + "\n")
            count += 1
    return count


def load_q_matroids(
    path,
    *,
    q: int,
    n: int,
) -> tuple[RankTableQMatroid, ...]:
    """Read rank tables written by :func:`save_q_matroids`."""
    with Path(path).open(encoding="ascii") as source:
        return tuple(
            RankTableQMatroid(q, n, tuple(map(int, line.split())))
            for line in source
        )


def _sage_matrix_from_gap(gap_matrix, q: int):
    """Convert a GAP finite-field matrix to the SageMath copy of ``GF(q)``."""
    F = GF(q)
    a = F.multiplicative_generator()
    gap_a = libgap.Z(q)

    return matrix(
        F,
        (
            tuple(
                F.zero()
                if bool(libgap.IsZero(x))
                else a ** int(libgap.LogFFE(x, gap_a))
                for x in row
            )
            for row in gap_matrix
        ),
    )


@cache
def _gap_linear_subspace_action(q: int, n: int):
    """Return GL, its action on all subspaces, and the action homomorphism."""
    V = VectorSpace(GF(q), n)
    subspaces = tuple(ReversedLexicographicOrderOfSubspaces(V).sorted())
    position = {X: i for i, X in enumerate(subspaces, start=1)}
    GLn = libgap.GL(n, q)
    generators = tuple(GLn.GeneratorsOfGroup())
    matrices = tuple(_sage_matrix_from_gap(g, q) for g in generators)
    permutations = tuple(
        libgap.PermList(
            tuple(position[X * g] for X in subspaces)
        )
        for g in matrices
    )
    action = libgap.Group(libgap.List(permutations))
    homomorphism = libgap.GroupHomomorphismByImages(
        GLn,
        action,
        libgap.List(generators),
        libgap.List(permutations),
    )
    return action, homomorphism


def _gap_projective_automorphism_group(
    M: RankTableQMatroid,
    tables: SubspaceTables,
):
    action, homomorphism = _gap_linear_subspace_action(M.q, M.n)
    bases = tuple(B + 1 for B in M.basis_ids(tables))
    stabilizer = libgap.Stabilizer(action, bases, libgap.OnSets)
    return stabilizer, homomorphism


def _projective_automorphism_group(
    M: RankTableQMatroid,
    tables: SubspaceTables,
):
    """Return the projective automorphism group acting on point IDs.

    The group is the stabilizer of the bases in the induced subspace action.
    The returned SageMath permutation group acts on the one-dimensional
    subspace IDs belonging to the q-matroid's ground space.
    """
    if M.n == 0:
        return PermutationGroup([])

    stabilizer, _ = _gap_projective_automorphism_group(M, tables)
    points = tables.contained_point_ids[M.ground_space_id(tables)]
    point_action = libgap.Action(
        stabilizer,
        tuple(e + 1 for e in points),
        libgap.OnPoints,
    )
    return PermutationGroup(gap_group=point_action, domain=points)


def _linear_automorphism_group(
    M: RankTableQMatroid,
    tables: SubspaceTables,
):
    """Return the linear automorphism group of a q-matroid.

    GAP computes the stabilizer of the bases in the induced action on
    subspaces, then takes its preimage in ``GL(n, q)``. Scalar matrices
    are retained.
    """
    if M.n == 0:
        return PermutationGroup([])

    stabilizer, homomorphism = _gap_projective_automorphism_group(M, tables)
    linear_stabilizer = libgap.PreImage(homomorphism, stabilizer)
    generators = tuple(
        _sage_matrix_from_gap(g, M.q)
        for g in linear_stabilizer.GeneratorsOfGroup()
    )
    if not generators:
        generators = (identity_matrix(GF(M.q), M.n),)
    return MatrixGroup(generators)


@cache
def linear_group_generator_permutations(
    q: int,
    n: int,
    k: int,
) -> tuple[tuple[int, ...], ...]:
    """Return the GL-generator action on ordered subspaces, using GAP indices."""
    if n == 0:
        return ()

    V = VectorSpace(GF(q), n)
    subspaces = tuple(ReversedLexicographicOrderOfSubspaces(V).sorted(k))
    position = {X: i for i, X in enumerate(subspaces, start=1)}

    return tuple(
        tuple(position[X * g.matrix()] for X in subspaces)
        for g in GL(n, q).gens()
    )


@cache
def _gap_subspace_action_group(q: int, n: int, k: int):
    load_result = libgap.LoadPackage("images", False)
    if load_result == libgap.fail:
        raise RuntimeError(
            "canonical testing requires the GAP package 'images'"
        )

    permutations = tuple(
        libgap.PermList(permutation)
        for permutation in linear_group_generator_permutations(
            q,
            n,
            k,
        )
    )
    return libgap.Group(libgap.List(permutations))


def _initialize_canonical_worker(
    tables: SubspaceTables,
    q: int,
    n: int,
    k: int,
) -> None:
    global _CANONICAL_WORKER_TABLES
    _CANONICAL_WORKER_TABLES = tables
    _gap_subspace_action_group.cache_clear()
    _gap_subspace_action_group(q, n, k)


def _worker_is_canonical(M: RankTableQMatroid) -> bool:
    return M.is_canonical(_CANONICAL_WORKER_TABLES)


def _batches(iterator, batch_size: int):
    iterator = iter(iterator)
    while batch := tuple(islice(iterator, batch_size)):
        yield batch


def _canonical_q_matroids(
    candidates: Iterator[RankTableQMatroid],
    tables: SubspaceTables,
    q: int,
    n: int,
    k: int,
    workers: int,
) -> Iterator[RankTableQMatroid]:
    if workers == 1:
        for M in candidates:
            if M.is_canonical(tables):
                yield M
        return

    context = multiprocessing.get_context("fork")
    batch_size = workers * 64
    with context.Pool(
        processes=workers,
        initializer=_initialize_canonical_worker,
        initargs=(tables, q, n, k),
    ) as pool:
        for batch in _batches(candidates, batch_size):
            chunksize = max(1, len(batch) // (workers * 4))
            flags = pool.map(
                _worker_is_canonical,
                batch,
                chunksize=chunksize,
            )
            yield from (
                M for M, canonical in zip(batch, flags) if canonical
            )


def enumerate_q_matroids(
    q: int,
    n: int,
    k: int,
    *,
    tables: SubspaceTables | None = None,
    workers: int = 1,
) -> Iterator[RankTableQMatroid]:
    """Yield canonical q-matroids by recursive one-dimensional extension.
    """
    if tables is None:
        tables = get_subspace_tables(q, n)

    classification: dict[tuple[int, int], tuple[RankTableQMatroid, ...]] = {}

    def representatives(i: int, j: int) -> tuple[RankTableQMatroid, ...]:
        if (i, j) not in classification:
            classification[i, j] = tuple(generate_representatives(i, j))
        return classification[i, j]

    def generate_representatives(i: int, j: int) -> Iterator[RankTableQMatroid]:
        if not 0 <= j <= i:
            return
        if i == 0:
            yield RankTableQMatroid(q, 0, (0,))
            return

        for M in representatives(i - 1, j - 1):
            yield M.trivial_extension(tables)
        candidates = (
            N
            for M in representatives(i - 1, j)
            for N in generate_one_dimensional_extensions(M, tables)
            if N.full_rank(tables) == j
        )
        yield from _canonical_q_matroids(
            candidates,
            tables,
            q,
            i,
            j,
            workers,
        )

    yield from generate_representatives(n, k)


__all__ = [
    "ModularCut",
    "ModularCutSelector",
    "RankTableQMatroid",
    "enumerate_q_matroids",
    "load_q_matroids",
    "save_q_matroids",
]
