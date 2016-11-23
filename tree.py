from collections import namedtuple
import ast
import cPickle

from grammar import *


class Tree(object):
    def __init__(self, type, label=None, children=None, holds_value=False):
        if isinstance(type, str):
            if type in ast_types:
                type = ast_types[type]

        self.type = type
        self.label = label
        self.holds_value = holds_value
        self.parent = None

        self.children = list()
        if children and isinstance(children, list):
            self.children.extend(children)
        elif children and isinstance(children, Tree):
            self.add_child(children)

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

    def add_child(self, child):
        child.parent = self
        self.children.append(child)

    def get_child_id(self, child):
        for i, _child in enumerate(self.children):
            if child == _child:
                return i

        raise KeyError

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

    def to_rule(self, include_terminal_val=True):
        src = Node(self.type, None)
        targets = []
        for child in self.children:
            if is_builtin_type(self.type) and not include_terminal_val:
                child_label = 'val'
            else:
                child_label = child.label

            tgt = Node(child.type, child_label)
            targets.append(tgt)

        rule = TypedRule(src, targets, tree=self)

        return rule

    def get_rule_list(self, parent_pos=-1, prev_pos=-1, include_leaf=True, leaf_val=False):
        """
        get the depth-first, left-to-right sequence of rule applications
        """
        if self.is_preterminal:
            if include_leaf:
                label = None
                if self.children[0].label is not None:
                    if leaf_val: label = self.children[0].label
                    else: label = 'val'

                return [TypedRule(Node(self.type, None), Node(self.children[0].type, label), tree=self)], \
                       [prev_pos + 1], [parent_pos]  # self.children[0].label
            return [], [], []
        elif self.is_leaf:
            return [], [], []

        src = Node(self.type, None)
        targets = []
        for child in self.children:
            tgt = Node(child.type, child.label)
            targets.append(tgt)

        rule = TypedRule(src, targets, tree=self)

        rule_list = [rule]
        rule_pos = prev_pos + 1
        rule_pos_list = [rule_pos]
        rule_par_pos_list = [parent_pos]

        cur_pos = rule_pos
        for child in self.children:
            child_rule_list, child_rule_pos, child_rule_par_pos = \
                child.get_rule_list(parent_pos=rule_pos, prev_pos=cur_pos, include_leaf=include_leaf, leaf_val=leaf_val)

            rule_list.extend(child_rule_list)
            rule_pos_list.extend(child_rule_pos)
            rule_par_pos_list.extend(child_rule_par_pos)

            cur_pos += len(child_rule_list)

        return rule_list, rule_pos_list, rule_par_pos_list

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
            new_tree.add_child(child.copy())

        return new_tree

    @property
    def size(self):
        if self.is_leaf:
            return 1

        node_num = 1
        for child in self.children:
            node_num += child.size

        return node_num

class DecodeTree(Tree):
    def __init__(self, type, label=None, children=None, holds_value=False, t=-1):
        super(DecodeTree, self).__init__(type, label, children, holds_value)

        # record the time step when this subtree is created from a rule application
        self.t = t

    def copy(self):
        new_tree = DecodeTree(self.type, self.label, holds_value=self.holds_value, t=self.t)
        if self.is_leaf:
            return new_tree

        for child in self.children:
            new_tree.add_child(child.copy())

        return new_tree

def add_root(tree):
    root_node = Tree('root')
    root_node.add_child(tree)

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
        parse_tree_rules, _, _ = parse_tree.get_rule_list(include_leaf=True)  # extract_rule(parse_tree)
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

    # print 'num. rules: %d' % len(rules)

    # get unique symbols, we only look at its type
    # symbol_types = set()
    # for rule in rules:
    #     symbol_types.add(rule.parent.type)
    #     for child in rule.children:
    #         symbol_types.add(child.type)
    #
    # print 'num. of symbol types: %d' % len(symbol_types)

    return grammar


if __name__ == '__main__':
    t = Tree('root', [
        Tree('a1', [Tree('a11', [Tree('a21')]), Tree('a12', [Tree('a21')])]),
        Tree('a2', [Tree('a21')])
    ])

    print t.__repr__()