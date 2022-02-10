from Classification import *

class Decision_Tree:
    # Each subtree is another DecisionTree, except leaf nodes which are Classification objects
    def __init__(self, children=None, attribute=None, label="ROOT-", threshold=None):
        self.children = children if children else set()
        self.attribute = attribute
        self.label = label
        self.threshold = threshold


    def getChildren(self):
        return self.children


    def getAttribute(self):
        return self.attribute


    def setAttribute(self, a):
        self.attribute = a


    def addChild(self, c):
        self.getChildren().add(c)


    def getLabel(self):
        return self.label


    def setLabel(self, l):
        self.label = l


    def getThreshold(self):
        return self.threshold


    def setThreshold(self, t):
        self.threshold = t


    def pretty_print_tree(self, node=None, prefix="", last=True):
        # This method is an adaptation of an algorithm from the following site:
        # https://vallentin.dev/2016/11/29/pretty-print-tree
        if node is None:
            node = self
        print(prefix, f"|-{node.getLabel()} {node.getThreshold()}- ", f'attr:{node.getAttribute().getName()}', sep="")
        prefix += "              " if last else "|             "
        child_count = len(node.children)
        for i, child in enumerate(node.getChildren()):
            last = i == (child_count - 1)
            if isinstance(child, Classification): 
                print(prefix, f"|-{child.getLabel()} {child.getThreshold()}- ", f"class={child.getClass()}", sep="")
                continue
            self.pretty_print_tree(child, prefix, last)


    def __str__(self):
        print("START ================")
        self.pretty_print_tree()
        print("STOP =================")        
        print()
        return ""