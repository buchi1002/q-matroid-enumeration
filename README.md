# qmatroid

## Purpose of this repository

1. Publish an example implementation of the enumeration algorithm in [the paper *On the one-dimensional extensions of q-matroids*](https://arxiv.org/abs/2503.06830)
2. Provide q-matroid classification results

## Environment

Linux or WSL2 is assumed. Other environments have not been tested. The SageMath version used for the computations is **10.6**.

After [setting up Conda with Miniforge or a similar distribution](https://doc.sagemath.org/html/en/installation/conda.html), create and activate an environment containing SageMath 10.6, GAP packages, and `psutil` for memory measurement.

```bash
conda create -n qmatroid -c conda-forge --override-channels sage=10.6 gap psutil
conda activate qmatroid
```

Next, install this repository's `qmatroid` package from the root directory of the repository (the directory containing `pyproject.toml`).

```bash
sage -pip install -e .
```

The `.` in `-e .` refers to the current directory. Following `pyproject.toml`, this command [installs `src/qmatroid` in editable mode](https://pip.pypa.io/en/stable/topics/local-project-installs/#editable-installs) in SageMath's Python environment, making it available in notebooks through `import qmatroid`.

The purpose of each package and how it is installed are listed below.

| Package | Purpose | Installation |
|---|---|---|
| SageMath (`sage.all`, `libgap`, etc.) and Jupyter | Computations with finite fields, subspaces, and groups; calls to GAP; notebook execution | `sage=10.6` in `conda create` |
| GAP packages (Images, etc.) | Testing canonical representatives, among other operations | `gap` in `conda create` |
| This repository's `qmatroid` | Enumerating, saving, and viewing q-matroids | `sage -pip install -e .` |
| `psutil` | Measuring memory usage in the classification notebook | `psutil` in `conda create` |

The GAP [Images](https://gap-packages.github.io/images/) package is used to test canonical representatives. See the [official documentation](https://doc.sagemath.org/html/en/reference/spkg/gap_packages.html) for installing GAP packages through Conda's `gap`. The following commands check that the SageMath version is 10.6 and that loading Images returns `true`.

```bash
sage --version
sage -c 'from sage.all import libgap; print(libgap.LoadPackage("images"))'
```

## Directory guide

### [`src/`](src/)

Example implementations of the q-matroid enumeration algorithm and tools for viewing classification results.

- [`qmatroid/finite_geometry.py`](src/qmatroid/finite_geometry.py): The subspace order from the paper and loading precomputed data for subspace containment, sums, and related operations.
- [`qmatroid/q_matroid_enumeration.py`](src/qmatroid/q_matroid_enumeration.py): Representing q-matroids by rank functions, enumerating modular cuts and one-dimensional extensions, testing canonical representatives, computing duals and automorphism groups, and saving and loading results.
- [`qmatroid/data/`](src/qmatroid/data/): Precomputed subspace data used for enumeration.

### [`notebooks/`](notebooks/)

Sample code for running the algorithm from the paper and viewing the results.

- [`run_classification.ipynb`](notebooks/run_classification.ipynb)
  - Classifies q-matroids in dimensions up to 5 over $\mathbb{F}_2$, or up to 4 over $\mathbb{F}_3$.
  - Saves the canonical representatives for each dimension and rank.
- [`view_classification_results.ipynb`](notebooks/view_classification_results.ipynb)
  - Displays the numbers of bases and hyperplanes, basis encodings, and automorphism groups of the saved canonical representatives.
- [`q_fano_candidates.ipynb`](notebooks/q_fano_candidates.ipynb)
  - Considers rank-3 canonical representatives on $\mathbb{F}_2^5$.
  - Extracts 10 candidates using the conditions on flats satisfied by restrictions of a q-Fano plane.

The bundled `results/` can be used directly to view the results. If the classification is rerun, enumerating rank-2 q-matroids on $\mathbb{F}_2^5$ and $\mathbb{F}_3^4$ may take several hours.

### [`results/`](results/)

The q-matroid classification results are stored as sequences of rank values, using the canonical representatives and subspace order from the paper. The included results cover:

- All ranks in dimensions 0 through 5 over $\mathbb{F}_2$
- All ranks in dimensions 0 through 4 over $\mathbb{F}_3$

Files are named `q{q}_dimension{n}_rank{k}.txt`, where $q$ is the field order, $n$ is the dimension of the ambient space, and $k$ is the rank of the q-matroid. Each line corresponds to the canonical representative of one isomorphism class. 

#### Subspace order

Following [Section 4.1 of the paper](https://arxiv.org/html/2503.06830#S4.SS1), the reverse canonical representation is obtained from a basis matrix $A$ of a subspace $X$. Reverse both the row and column orders, compute the reduced row echelon form, and reverse both orders again. In SageMath, this is:

```python
A[::-1, ::-1].echelon_form()[::-1, ::-1]
```

All subspaces are sorted in reverse lexicographic order by comparing the resulting matrices from the bottom row to the top, and from right to left within each row. If one sequence of rows being compared is a prefix of the other, the matrix with fewer rows comes first. The field elements are ordered as $0<1<g<g^2<\cdots<g^{q-2}$, where $g$ is the generator of the multiplicative group chosen by SageMath. For the bundled data, the orders are $0<1$ for $\mathbb{F}_2$ and $0<1<2$ for $\mathbb{F}_3$.

#### Storage format

Each line of a saved file records the canonical representative's **rank values on all subspaces**, separated by spaces. If all subspaces are listed in the order above as $X_0\prec\cdots\prec X_{N-1}$, a line contains the sequence

$$
r(X_0)\; r(X_1)\; \cdots\; r(X_{N-1}).
$$

The position in the sequence corresponds to the subspace ID, starting at 0. For example, the order for $\mathbb{F}_2^2$ is

$$
0\prec\langle(1,0)\rangle\prec\langle(0,1)\rangle
\prec\mathbb{F}_2^2\prec\langle(1,1)\rangle
$$

The first line, `0 0 1 1 1`, of [`q2_dimension2_rank1.txt`](results/q2_dimension2_rank1.txt) gives the rank values in this order. The basis encoding obtained from the values corresponding to one-dimensional subspaces is $(0,1,1)$.

**Notes**

- The data are not stored using the q-matroid encoding from the paper. This avoids the additional computation needed to reconstruct the rank function from the encoding.
- Within each file in [`results`](results), the q-matroids are stored in lexicographic order of their encodings.

## License

Except for `results/`, the contents of this repository are licensed under the [MIT License](LICENSE).

The classification data in `results/` are licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](results/LICENSE). When sharing the data, follow the license's attribution requirements and indicate any modifications.

If you use the classification data in academic work, please cite [the associated paper](https://arxiv.org/abs/2503.06830).
