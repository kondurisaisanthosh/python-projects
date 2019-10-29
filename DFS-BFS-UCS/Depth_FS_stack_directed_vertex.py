def DFS_vertexlist_Using_Stack(graph, begin, dest):              ###called method which returns path
    visitedvertex= set()
    stack1=[]
    stack1.append([begin])
    while stack1:
        path = stack1.pop()
        node = path[-1]                            ##assigns node variable with last element of the path
        if node not in visitedvertex:
            for i in reversed(graph[node]):
                alternate_path = list(path)
                alternate_path.append(i)
                stack1.append(alternate_path)
            visitedvertex.add(node)
        if node == dest:
            return path


def DFS_pathexpanded(elements,start,dest):                #method which returns states expanded
    stack1=[start]                                        #stack implementation
    path=[]
    while stack1:
        vertex=stack1.pop()
        if vertex in path:
            continue
        path.append(vertex)
        for neighbour in reversed(elements[vertex]):
            stack1.append(neighbour)
    return path                                            #returning states expanded



Elements_dic = {
    'S': ['d','e','p'],
    'a': [ ],
    'b': ['a'],
    'c': ['a'],
    'd': ['b','c','e'],
    'e': ['h','r'],
    'f':['c','G'],
    'G': [ ],
    'h': ['p', 'q'],
    'p': ['q'],
    'q': [],
    'r': ['f']
}

print("Paths Expanded=","->".join(DFS_pathexpanded(Elements_dic,'S','G')))   ##calling method which returns paths expanded
print("Path =","->".join(DFS_vertexlist_Using_Stack(Elements_dic, 'S', 'G')))##calling method that returns path

