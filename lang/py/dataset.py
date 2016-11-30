import ast

import astor

from lang.py.parse import parse, parse_tree_to_python_ast, canonicalize_code, get_grammar


def extract_grammar(code_file):
    line_num = 0
    parse_trees = []
    for line in open(code_file):
        code = line.strip()
        parse_tree = parse(code)

        # leaves = parse_tree.get_leaves()
        # for leaf in leaves:
        #     if not is_terminal_type(leaf.type):
        #         print parse_tree

        # parse_tree = add_root(parse_tree)

        parse_trees.append(parse_tree)

        # sanity check
        ast_tree = parse_tree_to_python_ast(parse_tree)
        ref_ast_tree = ast.parse(canonicalize_code(code)).body[0]
        source1 = astor.to_source(ast_tree)
        source2 = astor.to_source(ref_ast_tree)

        assert source1 == source2

        # check rules
        # rule_list = parse_tree.get_rule_list(include_leaf=True)
        # for rule in rule_list:
        #     if rule.parent.type == int and rule.children[0].type == int:
        #         # rule.parent.type == str and rule.children[0].type == str:
        #         pass

        # ast_tree = tree_to_ast(parse_tree)
        # print astor.to_source(ast_tree)
            # print parse_tree
        # except Exception as e:
        #     error_num += 1
        #     #pass
        #     #print e

        line_num += 1

    print 'total line of code: %d' % line_num

    grammar = get_grammar(parse_trees)

    with open('grammar.txt', 'w') as f:
        for rule in grammar:
            str = rule.__repr__()
            f.write(str + '\n')

    with open('parse_trees.txt', 'w') as f:
        for tree in parse_trees:
            f.write(tree.__repr__() + '\n')

    return grammar, parse_trees


def rule_vs_node_stat():
    line_num = 0
    parse_trees = []
    code_file = '/Users/yinpengcheng/Research/SemanticParsing/CodeGeneration/en-django/all.code'
    node_nums = rule_nums = 0.
    for line in open(code_file):
        code = line.strip()
        parse_tree = parse(code)
        node_nums += len(list(parse_tree.nodes))
        rules, _ = parse_tree.get_productions()
        rule_nums += len(rules)
        parse_trees.append(parse_tree)

        line_num += 1

    print 'avg. nums of nodes: %f' % (node_nums / line_num)
    print 'avg. nums of rules: %f' % (rule_nums / line_num)


if __name__ == '__main__':
    rule_vs_node_stat()