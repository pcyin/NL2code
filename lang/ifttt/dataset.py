from parse import ifttt_ast_to_parse_tree
from lang.grammar import Grammar
import logging

from nn.utils.generic_utils import init_logging

def load_examples(data_file):
    examples = []
    for line in open(data_file):
        d = line.strip().split('\t')
        description = d[4]
        code = d[9]
        parse_tree = ifttt_ast_to_parse_tree(code)

        examples.append({'description': description, 'parse_tree': parse_tree, 'code': code})

    return examples


def process_ifttt_dataset():
    data_file = '/Users/yinpengcheng/Research/SemanticParsing/ifttt/all.tsv'
    examples = load_examples(data_file)
    parse_trees = [e['parse_tree'] for e in examples]
    extract_grammar(parse_trees)


def extract_grammar(parse_trees):
    rules = set()

    for parse_tree in parse_trees:
        parse_tree_rules, rule_parents = parse_tree.get_productions()
        for rule in parse_tree_rules:
            rules.add(rule)

    rules = list(sorted(rules, key=lambda x: x.__repr__()))
    grammar = Grammar(rules)

    logging.info('num. rules: %d', len(rules))

    with open('grammar.txt', 'w') as f:
        for rule in grammar:
            str = rule.__repr__()
            f.write(str + '\n')

    with open('parse_trees.txt', 'w') as f:
        for tree in parse_trees:
            f.write(tree.__repr__() + '\n')

    return grammar


if __name__ == '__main__':
    init_logging('ifttt.log')
    process_ifttt_dataset()
