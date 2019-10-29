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
    'S': ['d','e','p'],                                       #elements in vertex form using dictionary in python
    'a': ['b', 'c'],
    'b': ['a', 'd'],
    'c': ['a','d','f'],
    'd': ['b','c','e','S'],
    'e': ['d', 'S', 'h','r'],
    'f':['c','G','r'],
    'G': ['f'],
    'h': ['p', 'q','e'],
    'p': ['S', 'h', 'q'],
    'q': ['p', 'h',],
    'r': ['e', 'f']
}


path=[]
print("States Expanded=","->".join(DFS_expanded_states_recursion('S',path)))  #recursive method which gives states expanded
DFS_vertexlist_Using_Recursion('S','G',['S'],[])                         #calling recursive method which returns shortest DFS path
