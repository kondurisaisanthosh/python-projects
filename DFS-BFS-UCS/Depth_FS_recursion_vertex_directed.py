def DFS_vertexlist_Using_Recursion(begin, dest,path,visited):              #called method which returns shortest path
    node = path[-1]                                                        #assigns node variable with last element of the path
    visited.append(node)
    if node == dest:
         print("Path=","->".join(path))
    else:
        for i in Elements_dic[node]:
            if i not in visited:
                alternate_path = list(path)
                alternate_path.append(i)
                DFS_vertexlist_Using_Recursion(begin, dest, alternate_path, visited)

def DFS_expanded_states_recursion(vertex, path):                     #Recursive method which returns states expanded
    path.append(vertex)
    for neighbour in Elements_dic[vertex]:
        if neighbour not in path:
            DFS_expanded_states_recursion(neighbour, path)      #Recursion
    return path                                                        #returning states expanded

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
    'q': [ ],
    'r': ['f']
}

path=[]
print("States Expanded=","->".join(DFS_expanded_states_recursion('S',path)))  #recursive method which gives states expanded
DFS_vertexlist_Using_Recursion('S','G',['S'],[])                         #calling recursive method which returns shortest DFS path
