import inspect, sys
import ast
from collections import OrderedDict, defaultdict

ast_types = dict()

# get all ast classes
for name, obj in inspect.getmembers(sys.modules['ast']):
    if inspect.isclass(obj):
        name = obj.__name__
        ast_types[name] = obj


terminal_ast_types = {
    ast.Pass,
    ast.Break,
    ast.Continue,
    ast.Add,
    ast.BitAnd,
    ast.BitOr,
    ast.BitXor,
    ast.Div,
    ast.FloorDiv,
    ast.LShift,
    ast.Mod,
    ast.Mult,
    ast.Pow,
    ast.Sub,
    ast.And,
    ast.Or,
    ast.Eq,
    ast.Gt,
    ast.GtE,
    ast.In,
    ast.Is,
    ast.IsNot,
    ast.Lt,
    ast.LtE,
    ast.NotEq,
    ast.NotIn,
    ast.Not,
    ast.USub
}


# x is a type
def typename(x):
    if isinstance(x, str):
        return x
    return x.__name__


def is_builtin_type(x):
    return x == str or x == int or x == float or x == bool or x == object or x == 'identifier'


def is_terminal_ast_type(x):
    if inspect.isclass(x) and x in terminal_ast_types:
        return True

    return False

def is_terminal_type(x):
    if is_builtin_type(x):
        return True

    if x == 'epsilon':
        return True

    if inspect.isclass(x) and (issubclass(x, ast.Pass) or issubclass(x, ast.Raise) or issubclass(x, ast.Break)
                               or issubclass(x, ast.Continue)
                               or issubclass(x, ast.Return)
                               or issubclass(x, ast.operator) or issubclass(x, ast.boolop)
                               or issubclass(x, ast.Ellipsis) or issubclass(x, ast.unaryop)
                               or issubclass(x, ast.cmpop)):
        return True

    return False


class Node:
    def __init__(self, node_type, label):
        self.type = node_type
        self.label = label

    @property
    def is_preterminal(self):
        return is_terminal_type(self.type)

    def __eq__(self, other):
        return self.type == other.type and self.label == other.label

    def __hash__(self):
        return typename(self.type).__hash__() ^ self.label.__hash__()

    def __repr__(self):
        repr_str = typename(self.type)
        if self.label:
            repr_str += '{%s}' % self.label
        return repr_str


class TypedRule:
    def __init__(self, parent, children, tree=None):
        self.parent = parent
        if isinstance(children, list) or isinstance(children, tuple):
            self.children = tuple(children)
        else:
            self.children = (children, )

        # tree property is not incorporated in eq, hash
        self.tree = tree

    # @property
    # def is_terminal_rule(self):
    #     return is_terminal_type(self.parent.type)

    def __eq__(self, other):
        return self.parent == other.parent and self.children == other.children

    def __hash__(self):
        return self.parent.__hash__() ^ self.children.__hash__()

    def __repr__(self):
        return '%s -> %s' % (self.parent, ', '.join([c.__repr__() for c in self.children]))

root_ast_types = {ast.Module, ast.ClassDef, ast.stmt,
                  ast.expr, ast.FunctionDef, ast.ClassDef, ast.Return,
                  ast.Delete, ast.Assign, ast.AugAssign, ast.Print,
                  ast.For, ast.While, ast.If, ast.With, ast.Raise,
                  ast.TryExcept, ast.TryFinally, ast.Assert, ast.Import,
                  ast.ImportFrom, ast.Exec, ast.Global, ast.Expr, ast.Pass}


class Grammar:
    def __init__(self, rules):
        self.rules = rules
        self.index = defaultdict(list)
        self.rule_to_id = OrderedDict()

        node_types = set()
        for rule in self.rules:
            self.index[rule.parent].append(rule)

            # we also store all unique node types
            node_types.add(str(rule.parent.type))
            for child in rule.children:
                node_types.add(str(child.type))

        self.node_types = dict()
        for i, type in enumerate(node_types, start=0):
            self.node_types[type] = i

        for gid, rule in enumerate(rules, start=0):
            self.rule_to_id[rule] = gid

        self.id_to_rule = dict((v, k) for (k, v) in self.rule_to_id.iteritems())

        self.root_rules = [rule for rule in rules if rule.parent.type in root_ast_types]

    def __iter__(self):
        return self.rules.__iter__()

    def __getitem__(self, lhs):
        # if lhs.type == 'root':
        #    return self.root_rules

        key_node = Node(lhs.type, None)  # Rules are indexed by types only
        if key_node in self.index:
            return self.index[key_node]

        raise KeyError('key=%s' % key_node)

    def get_node_type_id(self, node):
        from tree import Tree

        if isinstance(node, Node) or isinstance(node, Tree):
            type_repr = str(node.type)
            return self.node_types[type_repr]
        else:
            # assert isinstance(node, str)
            # it is a type
            type_repr = str(node)
            return self.node_types[type_repr]
