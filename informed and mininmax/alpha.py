class BinaryTree():
    def __init__(self, rootid, Required_value=None):
        self.left = None
        self.right = None
        self.rootid = rootid
        if Required_value is None:
            self.alternateRequired_value()
        else:
            self.Required_value = Required_value

    def alternateRequired_value(self):
        self.Required_value = 0

    def getLeftChild(self):
        return self.left

    def getRightChild(self):
        return self.right

    def setNodeValue(self, Required_value):
        self.Required_value = Required_value

    def getNodeValue(self):
        return self.Required_value

    def getNodeId(self):
        return self.rootid

    def insertRight(self, rootid, newNode=None):
        if self.right == None:
            self.right = BinaryTree(rootid, newNode)
        else:
            tree = BinaryTree(rootid, newNode)
            tree.right = self.right
            self.right = tree

    def insertLeft(self, rootid, newNode=None):
        if self.left == None:
            self.left = BinaryTree(rootid, newNode)
        else:
            tree = BinaryTree(rootid, newNode)
            tree.left = self.left
            self.left = tree


alphapath=[]

def printTree(tree):                 #prints tree
        if tree != None:
            printTree(tree.getLeftChild())
            print(tree.getNodeValue())
            printTree(tree.getRightChild())


def getTree():                     #tree Data
    head = BinaryTree("head")
    head.insertLeft("HL")
    head.insertRight("HR")
    HL = head.getLeftChild()
    HL.insertLeft("HLL")
    HL.insertRight("HLR")
    HLL = HL.getLeftChild()
    HLL.insertLeft("HLLL")
    HLL.insertRight("HLLR")
    HLR = HL.getRightChild()
    HLR.insertLeft("HLRL")
    HLR.insertRight("HLRR")
    HLLL = HLL.getLeftChild()
    HLLL.insertLeft("HLLLL")
    HLLLL = HLLL.getLeftChild()
    HLLLL.setNodeValue(3)
    HLLL.insertRight("HLLLR")
    HLLLR = HLLL.getRightChild()
    HLLLR.setNodeValue(10)
    HLLR = HLL.getRightChild()
    HLLR.insertLeft("HLLRL")
    HLLRL = HLLR.getLeftChild()
    HLLRL.setNodeValue(2)
    HLLR.insertRight("HLLRR")
    HLLRR = HLLR.getRightChild()
    HLLRR.setNodeValue(9)
    HLRL = HLR.getLeftChild()
    HLRL.insertLeft("HLRLL")
    HLRLL = HLRL.getLeftChild()
    HLRLL.setNodeValue(10)
    HLRL.insertRight("HLRLR")
    HLRLR = HLRL.getRightChild()
    HLRLR.setNodeValue(7)
    HLRR = HLR.getRightChild()
    HLRR.insertLeft("HLRRL")
    HLRRL = HLRR.getLeftChild()
    HLRRL.setNodeValue(5)
    HLRR.insertRight("HLRRR")
    HLRRR = HLRR.getRightChild()
    HLRRR.setNodeValue(9)

    HR = head.getRightChild()
    HR.insertLeft("HRL")
    HR.insertRight("HRR")
    HRL = HR.getLeftChild()
    HRL.insertLeft("HRLL")
    HRL.insertRight("HRLR")
    HRR = HR.getRightChild()
    HRR.insertLeft("HRRL")
    HRR.insertRight("HRRR")
    HRLL = HRL.getLeftChild()
    HRLL.insertLeft("HRLLL")
    HRLLL = HRLL.getLeftChild()
    HRLLL.setNodeValue(2)
    HRLL.insertRight("HLLLL0")
    HLLLL0 = HRLL.getRightChild()
    HLLLL0.setNodeValue(5)
    HRLR = HRL.getRightChild()
    HRLR.insertLeft("HLLLL1")
    HLLLL1 = HRLR.getLeftChild()
    HLLLL1.setNodeValue(6)
    HRLR.insertRight("HLLLL2")
    HLLLL2 = HRLR.getRightChild()
    HLLLL2.setNodeValue(4)
    HRRL = HRR.getLeftChild()
    HRRL.insertLeft("HLLLL3")
    HLLLL3 = HRRL.getLeftChild()
    HLLLL3.setNodeValue(2)
    HRRL.insertRight("HLLLL4")
    HLLLL4 = HRRL.getRightChild()
    HLLLL4.setNodeValue(7)
    HRRR = HRR.getRightChild()
    HRRR.insertLeft("HLLLL5")
    HLLLL5 = HRRR.getLeftChild()
    HLLLL5.setNodeValue(9)
    HRRR.insertRight("HLLLL6")
    HLLLL6 = HRRR.getRightChild()
    HLLLL6.setNodeValue(1)

    return head

def pruningAlgorithm(current, tree, deep,chance ,al, be):
    if tree.left is None and tree.right is None:
        return tree.getNodeValue()

    if (chance):
        actual = -10000
        for i in range(2):
            if i == 0:
                actual = max(actual, pruningAlgorithm(current + 1, tree.getLeftChild(), deep, False, al, be)) #calling recursive function
            else:
                actual = max(actual, pruningAlgorithm(current + 1, tree.getRightChild(), deep, False, al, be))

            al = max(actual, al)
            tree.setNodeValue(actual)
            alphapath.append(tree.getNodeId())

            if al >= be:                   #pruning out the value
                tree.setNodeValue(0)
                break
        return actual
    else:
        actual = 10000
        for i in range(2):
            if i == 0:
                actual = min(actual, pruningAlgorithm(current + 1, tree.getLeftChild(), deep, True, al, be))
            else:
                actual = min(actual, pruningAlgorithm(current + 1, tree.getRightChild(), deep, True, al, be))
            be = min(actual, be)
            tree.setNodeValue(actual)
            alphapath.append(tree.getNodeId())
            if be <= al:
                tree.setNodeValue(0)
                break
        return actual

    return actual


def alphaBeta(tree):
    actual = -10000
    for i in range(2):
        if i == 0:
            data = pruningAlgorithm(0, tree.getLeftChild(), 5, False, -10000, 10000)
        else:
            data = pruningAlgorithm(0, tree.getRightChild(), 5, False, -10000, 10000)

        if data > actual:
            actual = data
    tree.setNodeValue(actual)
    return actual


tree = getTree()
b = alphaBeta(tree)

print("AlphaBeta value is : " + str(b))

