from collections import namedtuple


class Tree:
    def __init__(self, name, children=None):
        self.name = name
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

    def __repr__(self):
        repr_str = ''
        if not self.is_leaf:
            repr_str += '('

        repr_str += str(self.name)

        if not self.is_leaf:
            for child in self.children:
                repr_str += ' ' + child.__repr__()
            repr_str += ')'

        return repr_str

    def get_rule_list(self, include_leaf=True):
        if self.is_preterminal:
            if include_leaf:
                return [Rule(self.name, self.children[0].name)]
            return []
        elif self.is_leaf:
            return []

        targets = [child.name for child in self.children]
        rule = Rule(self.name, targets)

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

    rules.add(Rule(parse_tree.name, [child.name for child in parse_tree.children]))

    for child in parse_tree.children:
        child_rules = extract_rule(child)
        for rule in child_rules:
            rules.add(rule)

    return rules


def get_grammar(parse_trees):
    grammar = set()
    for parse_tree in parse_trees:
        rules = extract_rule(parse_tree)
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