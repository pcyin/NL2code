# x is a type
def typename(x):
    if isinstance(x, str):
        return x
    return x.__name__

def escape(text):
    text = text \
        .replace('"', '-``-') \
        .replace('\'', '-`-') \
        .replace(' ', '-SP-') \
        .replace('\t', '-TAB-') \
        .replace('\n', '-NL-') \
        .replace('(', '-LRB-') \
        .replace(')', '-RRB-') \
        .replace('|', '-BAR-')
    return repr(text)[1:-1] if text else '-NONE-'

def unescape(text):
    text = text \
        .replace('-``-', '"') \
        .replace('-`-', '\'') \
        .replace('-SP-', ' ') \
        .replace('-TAB-', '\\t') \
        .replace('-NL-', '\\n') \
        .replace('-LRB-', '(') \
        .replace('-RRB-', ')') \
        .replace('-BAR-', '|') \
        .replace('-NONE-', '')

    return text