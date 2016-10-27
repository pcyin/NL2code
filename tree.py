from collections import namedtuple
import ast

from grammar import *


class Tree:
    def __init__(self, type, label=None, children=None):
        if isinstance(type, str):
            if type in ast_types:
                type = ast_types[type]

        self.type = type
        self.label = label

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

    def __repr__(self):
        repr_str = ''
        if not self.is_leaf:
            repr_str += '('

        repr_str += self.type_name

        if self.label:
            repr_str += '{%s}' % self.label

        if not self.is_leaf:
            for child in self.children:
                repr_str += ' ' + child.__repr__()
            repr_str += ')'

        return repr_str

    def get_rule_list(self, include_leaf=True):
        if self.is_preterminal:
            if include_leaf:
                return [TypedRule(Node(self.type, self.label), Node(self.children[0].type, self.children[0].label))]
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
            rule_list.extend(child.get_rule_list(include_leaf=include_leaf))

        return rule_list


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
    grammar = set()
    for parse_tree in parse_trees:
        rules = parse_tree.get_rule_list(include_leaf=False)  # extract_rule(parse_tree)
        for rule in rules:
            grammar.add(rule)

    print 'num. rules: %d' % len(grammar)

    # get unique symbols
    symbols = set()
    for rule in grammar:
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