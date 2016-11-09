from collections import namedtuple
import ast
import cPickle

from grammar import *


class Tree:
    def __init__(self, type, label=None, children=None, holds_value=False):
        if isinstance(type, str):
            if type in ast_types:
                type = ast_types[type]

        self.type = type
        self.label = label
        self.holds_value = holds_value

        self.children = list()
        if children and isinstance(children, list):
            self.children.extend(children)
        elif children and isinstance(children, Tree):
            self.children.append(children)

    @property
    def is_leaf(self):
        return len(self.children) == 0

    @property
    def is_preterminal(self):
        return len(self.children) == 1 and self.children[0].is_leaf

    @property
    def type_name(self):
        if isinstance(self.type, str):
            return self.type
        return self.type.__name__

    @property
    def node(self):
        return Node(self.type, self.label)

    def apply_rule(self, rule):
        assert rule.parent.type == self.type

        for child_node in rule.children:
            child = Tree(child_node.type, child_node.label)
            if is_builtin_type(rule.parent.type):
                child.label = None
                child.holds_value = True

            self.children.append(child)

    def append_token(self, token):
        if self.label is None:
            self.label = token
        else:
            self.label += token

    def __repr__(self):
        repr_str = ''
        if not self.is_leaf:
            repr_str += '('

        repr_str += self.type_name

        if self.label is not None:
            repr_str += '{%s}' % self.label

        if not self.is_leaf:
            for child in self.children:
                repr_str += ' ' + child.__repr__()
            repr_str += ')'

        return repr_str

    def get_leaves(self):
        if self.is_leaf:
            return [self]

        leaves = []
        for child in self.children:
            leaves.extend(child.get_leaves())

        return leaves

    def get_rule_list(self, include_leaf=True, leaf_val=False):
        if self.is_preterminal:
            if include_leaf:
                label = None
                if self.children[0].label is not None:
                    if leaf_val: label = self.children[0].label
                    else: label = 'val'

                return [TypedRule(Node(self.type, None), Node(self.children[0].type, label))]  # self.children[0].label
            return []
        elif self.is_leaf:
            return []

        src = Node(self.type, None)
        targets = []
        for child in self.children:
            tgt = Node(child.type, child.label)
            targets.append(tgt)

        rule = TypedRule(src, targets)

        rule_list = [rule]
        for child in self.children:
            rule_list.extend(child.get_rule_list(include_leaf=include_leaf, leaf_val=leaf_val))

        return rule_list

    def copy(self):
        # if not hasattr(self, '_dump'):
        #     dump = cPickle.dumps(self, -1)
        #     setattr(self, '_dump', dump)
        #
        #     return cPickle.loads(dump)
        #
        # return cPickle.loads(self._dump)

        new_tree = Tree(self.type, self.label, holds_value=self.holds_value)
        if self.is_leaf:
            return new_tree

        for child in self.children:
            new_tree.children.append(child.copy())

        return new_tree

    @property
    def size(self):
        if self.is_leaf:
            return 1

        node_num = 1
        for child in self.children:
            node_num += child.size

        return node_num

def add_root(tree):
    root_node = Tree('root')
    root_node.children.append(tree)

    return root_node


class Rule:
    def __init__(self, parent, children):
        self.parent = parent
        if isinstance(children, list) or isinstance(children, tuple):
            self.children = tuple(children)
        else:
            self.children = (children, )

    def __eq__(self, other):
        return self.parent == other.parent and self.children == other.children

    def __hash__(self):
        return self.parent.__hash__() ^ self.children.__hash__()

    def __repr__(self):
        return '%s -> %s' % (self.parent, ', '.join([str(c) for c in self.children]))


# Rule = namedtuple('Rule', ['parent', 'children'])


def extract_rule(parse_tree):
    rules = set()

    if parse_tree.is_leaf or parse_tree.is_preterminal:
        return rules

    rule_src = parse_tree.type_name
    targets = []
    for child in parse_tree.children:
        tgt = child.type_name
        if child.label:
            tgt += '{%s}' % child.label

        targets.append(tgt)

    rules.add(Rule(rule_src, targets))

    for child in parse_tree.children:
        child_rules = extract_rule(child)
        for rule in child_rules:
            rules.add(rule)

    return rules


def get_grammar(parse_trees):
    rules = set()
    # rule_num_dist = defaultdict(int)

    for parse_tree in parse_trees:
        parse_tree_rules = parse_tree.get_rule_list(include_leaf=True)  # extract_rule(parse_tree)
        # len_span = len(parse_tree_rules) / 10
        # rule_num_dist['%d ~ %d' % (10 * len_span, 10 * len_span + 10)] += 1
        for rule in parse_tree_rules:
            rules.add(rule)

    # sorted_rule_num = sorted(rule_num_dist, reverse=False)
    # N = sum(rule_num_dist.itervalues())
    # print 'rule num distribution'
    # for k, v in rule_num_dist.iteritems():
    #     print k, v / float(N)

    rules = list(sorted(rules, key=lambda x: x.__repr__()))
    grammar = Grammar(rules)

    print 'num. rules: %d' % len(rules)

    # get unique symbols
    symbols = set()
    for rule in rules:
        symbols.add(rule.parent)
        for child in rule.children:
            symbols.add(child)

    print 'num. of symbols: %d' % len(symbols)

    return grammar


if __name__ == '__main__':
    t = Tree('root', [
        Tree('a1', [Tree('a11', [Tree('a21')]), Tree('a12', [Tree('a21')])]),
        Tree('a2', [Tree('a21')])
    ])

    print t.__repr__()