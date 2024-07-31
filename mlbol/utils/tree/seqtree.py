from typing import Iterable, Iterator, TypeVar

T = TypeVar("T")

PIPE = "│"
ELBOW = "└──"
TEE = "├──"
PIPE_PREFIX = "│   "
SPACE_PREFIX = "    "


class Node:
    def __init__(self, value, depth: int, category: str):
        self.children = []
        self.value = value
        self.depth = depth
        self.category = category

    def __repr__(self):
        return f"Node(value={self.value}, depth={self.depth}, category={self.category})"

    def add_child(self, node):
        self.children.append(node)

    def __iter__(self):
        return iter(self.children)

    def dfs(self):
        yield self
        for child in self:
            yield from child.dfs()


class SequenceTree(Iterable[T]):
    def __init__(self, seq: Iterable[T]):
        self._root = None
        self._tree = []
        self._build_tree_root(seq)
        self._build_tree_body(seq, 1, self._root)

    def __iter__(self) -> Iterator[T]:
        return self._root.dfs()

    def draw(self):
        for node in self._tree:
            print(node)

    def _is_internal(self, seq: T) -> bool:
        return isinstance(seq, (list, tuple))

    def _build_tree_root(self, seq: T):
        self._root = Node(type(seq).__name__, 0, "root")
        self._tree.append(f"{self._root}")
        if self._is_internal(seq) and len(seq) > 0:
            self._tree.append(PIPE)

    def _build_tree_body(self, seq: T, depth: int, parent: Node, prefix: str = ""):
        els = sorted(iter(seq), key=lambda el: ~self._is_internal(el))
        num_els = len(els)
        for id_el, el in enumerate(els):
            connector = ELBOW if id_el == num_els - 1 else TEE
            if self._is_internal(el):
                self._add_internal(el, depth, parent, id_el, num_els, prefix, connector)
            else:
                self._add_leaf(el, depth, parent, prefix, connector)

    def _add_internal(
        self,
        seq: T,
        depth: int,
        parent: Node,
        id_el: int,
        num_els: int,
        prefix: str,
        connector: str,
    ):
        node = Node(type(seq).__name__, depth, "internal")
        parent.add_child(node)
        self._tree.append(f"{prefix}{connector} {node}")
        if id_el != num_els - 1:
            prefix += PIPE_PREFIX
        else:
            prefix += SPACE_PREFIX
        self._build_tree_body(seq, depth + 1, node, prefix)

    def _add_leaf(self, seq: T, depth: int, parent: Node, prefix: str, connector: str):
        node = Node(seq, depth, "leaf")
        parent.add_child(node)
        self._tree.append(f"{prefix}{connector} {node}")


def seqtree(seq: Iterable) -> None:
    tree = SequenceTree(seq)
    tree.draw()