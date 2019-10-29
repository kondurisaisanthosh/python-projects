MAX, MIN = 100000, -1000000

class BinaryTree():

    def __init__(self,rootid):                   #constructor
      self.left = None
      self.right = None
      self.rootid = rootid
      self.value = None

    def getLeftChild(self):                 #gets left child
        return self.left
    def getRightChild(self):                #gets right child
        return self.right

    def setNodeValue(self,value):          #sets node value
        self.value = value
    def getNodeValue(self):                #gives node value
        return self.value

    def insertRight(self,newNode):         #inserts child to right
        if self.right == None:
            self.right = BinaryTree(newNode)
        else:
            tree = BinaryTree(newNode)
            tree.right = self.right
            self.right = tree

    def insertLeft(self,newNode):           #inserts child to left
        if self.left == None:
            self.left = BinaryTree(newNode)
        else:
            tree = BinaryTree(newNode)
            tree.left = self.left
            self.left = tree


def minimax(depth, nodeIndex, maximizingPlayer,values, alpha, beta):

    if depth == 3:
        return values[nodeIndex]

    if maximizingPlayer:
        best = MIN
        for i in range(0, 2):
            val = minimax(depth + 1, nodeIndex * 2 + i,False, values, alpha, beta)
            best = max(best, val)
            alpha = max(alpha, best)
            if beta <= alpha:
                break
        return best
    else:
        best = MAX
        for i in range(0, 2):
            val = minimax(depth + 1, nodeIndex * 2 + i,True, values, alpha, beta)
            best = min(best, val)
            beta = min(beta, best)
            if beta <= alpha:
                break
        return best



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


print("The optimal value is :", minimax(0, 0, True, head, MIN, MAX))

