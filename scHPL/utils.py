import re
import pathlib
import copy
import numpy as np
from newick import Node
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

COMMENT = re.compile(r'\[[^\]]*\]')


class TreeNode(Node):
    '''
    This class is an extension of Node, but with the possibility
    to add a classifier
    '''
    
    def __init__(self, name=None, **kw):
        super().__init__(name=name, **kw)
        self.classifier = None
        self.dimred = False
        self.pca = None
        self.explainingpc = None
        self.maxRE = 0 #the highest reconstruction error on the training data

    @classmethod
    def create(cls, name=None, descendants=None, **kw):
        """
        Create a new `Node` object.

        :param name: Node label.
        :param descendants: list of descendants or `None`.
        :param kw: Additonal keyword arguments are passed through to `Node.__init__`.
        :return: `Node` instance.
        """
        node = cls(name=name, **kw)
        for descendant in descendants or []:
            node.add_descendant(descendant)
        return node
    
    def get_leaf_names_first(self):
        """
        Get the names of all the leaf nodes of the subtree descending from
        this node.
        :return: List of names of Nodes with no descendants.
        """
        return [n.name[0] for n in self.get_leaves()]


    
    def set_classifier(self, classifier):
        """
        Add a classifier to the node.
        """
        self.classifier = copy.deepcopy(classifier)
    
    def get_classifier(self):
        return self.classifier
    
    def set_dimred(self, value):
        self.dimred = value
        
    def get_dimred(self):
        return self.dimred
        
    def get_pca(self):
        return copy.deepcopy(self.pca), copy.deepcopy(self.explainingpc)
    
    def set_pca(self, pca, explainingpc):
        self.pca = copy.deepcopy(pca)
        self.explainingpc = copy.deepcopy(explainingpc)
        
    def set_RE(self, maxRE):
        self.maxRE = maxRE
        
    def get_RE(self):
        return self.maxRE

def read_tree(fname, encoding='utf8', strip_comments=False, **kw):
    """
    Load a list of trees from a Newick formatted file.

    :param fname: file path.
    :param strip_comments: Flag signaling whether to strip comments enclosed in square \
    brackets.
    :param kw: Keyword arguments are passed through to `Node.create`.
    :return: List of Node objects.
    """
    kw['strip_comments'] = strip_comments
    with pathlib.Path(fname).open(encoding=encoding) as fp:
        return load(fp, **kw)

def load(fp, strip_comments=False, **kw):
    """
    Load a list of trees from an open Newick formatted file.

    :param fp: open file handle.
    :param strip_comments: Flag signaling whether to strip comments enclosed in square \
    brackets.
    :param kw: Keyword arguments are passed through to `Node.create`.
    :return: List of Node objects.
    """
    kw['strip_comments'] = strip_comments
    return create_tree(fp.read(), **kw)


def create_tree(s, strip_comments=False, **kw):
    """
    Load a list of trees from a Newick formatted string.

    :param s: Newick formatted string.
    :param strip_comments: Flag signaling whether to strip comments enclosed in square \
    brackets.
    :param kw: Keyword arguments are passed through to `Node.create`.
    :return: List of Node objects.
    """
    kw['strip_comments'] = strip_comments
    return [_parse_node(ss.strip(), **kw) for ss in s.split(';') if ss.strip()]
    
def _parse_node(s, strip_comments=False, **kw):
    """
    Parse a Newick formatted string into a `Node` object.

    :param s: Newick formatted string to parse.
    :param strip_comments: Flag signaling whether to strip comments enclosed in square \
    brackets.
    :param kw: Keyword arguments are passed through to `Node.create`.
    :return: `Node` instance.
    """
    if strip_comments:
        s = COMMENT.sub('', s)
    s = s.strip()
    parts = s.split(')')
    if len(parts) == 1:
        descendants, label = [], s
    else:
        if not parts[0].startswith('('):
            raise ValueError('unmatched braces %s' % parts[0][:100])
        descendants = list(_parse_siblings(')'.join(parts[:-1])[1:], **kw))
        label = parts[-1]
    name, length = _parse_name_and_length(label)
    return TreeNode.create(name=[name], descendants=descendants, **kw)

def _parse_name_and_length(s):
    length = None
    if ':' in s:
        s, length = s.split(':', 1)
    return s or None, length or None

def _parse_siblings(s, **kw):
    """
    http://stackoverflow.com/a/26809037
    """
    bracket_level = 0
    current = []

    # trick to remove special-case of trailing chars
    for c in (s + ","):
        if c == "," and bracket_level == 0:
            yield _parse_node("".join(current), **kw)
            current = []
        else:
            if c == "(":
                bracket_level += 1
            elif c == ")":
                bracket_level -= 1
            current.append(c)
            

def rename_node(old_name, new_name, tree):
    '''
    Function to rename a node in the tree manually.
    
    Parameters:
    ----------
    old_name: old name of the node
    new_name: new_name of the node
    tree: tree containing the node
    
    Return:
    -------
    tree: updated tree
    '''
    
    for n in tree[0].walk(mode = 'postorder'):
        if(np.isin(old_name, n.name)):
            n.name = [new_name]
            return tree
    
    print('Node not found, node could not be renamed.')
    
    return tree

def remove_node(name, tree, children = True):
    '''
    

    Parameters
    ----------
    name: name of the node to remove
    tree : tree containing this node
    children : if True: children are also removed, if False: children are
    rewired to parent node 
    
    Returns
    -------
    tree: updated tree

    '''
    
   
    if name == 'root':
        print('Root cannot be removed from the tree.')
        return tree
    else:
        for n in tree[0].walk(mode = 'postorder'):
            if(np.isin(name, n.name)):
                parentnode = n.ancestor
                
                if children == False:
                    for c in n.descendants:
                        parentnode.add_descendant(c)
                        c.ancestor = parentnode
                
                old_descendants = parentnode.descendants
                #remove children from the previous parent
                parentnode.descendants = []
                for j in old_descendants:
                    if np.isin(name, j.name, invert=True):
                        parentnode.add_descendant(j)
                return tree
    
    print('Node not found, node could not be deleted.')
    
    
    
    return tree

def add_node(name, tree, parent, children = None):
    '''
    Function to add a node to the tree manually.
    
    Parameters:
    ----------
    name: name of the cell population to add
    tree: tree the node should be added to
    parent: parent node the cell population should be added to. If the node
    should be added to the root, this should be 'root'
    children: if the new node is an internal node, the child nodes should 
    also be defined. Format should be a string vector e.g. ['child1', 'child2']
    
    Return:
    -------
    tree: updated tree
    '''
    
    newnode = TreeNode([name])
    
    if parent == 'root':
        tree[0].add_descendant(newnode)
        if children != None:
            _rewire_children(newnode, tree[0], children)
        return tree
    else:
        for n in tree[0].walk(mode = 'postorder'):
            if(np.isin(parent, n.name)):
                n.add_descendant(newnode)
                if children != None: 
                    _rewire_children(newnode, n, children)
                return tree
    
    print('Parent node not found, node could not be added.')
    
    
    return tree

def _rewire_children(newnode, parentnode, children):
    '''
    Function to rewire an existing node to a new ancestor
    
    Parameters
    ----------
    newnode: the new ancestor of children
    parentnode: the old ancestor of children
    children: nodes that should be rewired
    '''
    
    #add children to the newnode
    for i in parentnode.descendants:
        if np.isin(i.name, children).any():
            i.ancestor = newnode 
            newnode.add_descendant(i) 

    
    #remove children from the previous parent
    olddescendants = parentnode.descendants
    parentnode.descendants = []
    for j in olddescendants:
        if np.isin(j.name, children).any() == False:
            parentnode.add_descendant(j)
    
    return 

def print_tree(tree):
    '''
    Print a tree

    Parameters
    ----------
    tree : tree to print

    Returns
    -------
    None.

    '''
    
    global ver
    ver = 0.93
    
    count = _count_nodes(tree)
    ver_steps = 0.9/count
    plot_height = count*0.3
    print(plot_height)
    fig = plt.figure(figsize=(6,plot_height))
        
    _print_node(tree[0], hor=0.05, ver_steps=ver_steps, fig=fig)
        
    plt.show()
    

def _print_node(node, hor, ver_steps, fig):
    global ver
#     hor2 = hor
    txt = node.name[0]
    for n in node.name:
        if(n != txt):
            txt = txt + ' & ' + n
        
    # Add horizontal line
    x, y = ([np.max([0.05, hor-0.045]), hor], [ver, ver])
    line = mlines.Line2D(x,y, lw=1)
    fig.add_artist(line)

    # Add textbox
    fig.text(hor,ver, txt, size=10,
             ha = 'left', va='center',
             bbox = dict(boxstyle='round', fc='w', ec='k'))

    hor = hor+0.05
    
    ver_line_start = ver
    ver_line_end = ver
    
    for i in node.descendants:
        ver = ver-ver_steps
        ver_line_end = ver
        _print_node(i, hor, ver_steps, fig)
        
    # Add vertical line
    x, y = ([np.max([0.05, hor-0.045]), np.max([0.05, hor-0.045])], 
            [ver_line_start, ver_line_end])
    line = mlines.Line2D(x,y, lw=1)
    fig.add_artist(line)
        
def _count_nodes(tree):
    
    count = 0
    for n in tree[0].walk():
        count=count+1
        
    return count


    

    


    