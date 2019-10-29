def DFS_expanded_states_recursion(vertex, path):                     #Recursive method which returns states expanded
    path.append(vertex)
    neighbour_mat=[]
    for key,value in elements.items():
        if value==vertex:
            neighbour_mat=Elements_matrix[key]
    i=0
    while i<len(neighbour_mat):
        if neighbour_mat[i]==1:
            if elements[i] not in path:
                 DFS_expanded_states_recursion(elements[i], path)      #Recursion
        i+=1
    return path                                                          #returning states expanded


def DFS_adjacencymatrix_Using_Recursion(begin, dest,path,visited):              #called method which returns shortest path
    node = path[-1]                                                        #assigns node variable with last element of the path
    visited.append(node)
    if node == dest:
         print("Path=","->".join(path))
    else:
        adj_mat=[]
        for key,value in elements.items():
            if value==node:
                adj_mat=Elements_matrix[key]
        index=0
        while index<len(adj_mat):
            if adj_mat[index]==1 and elements[index] not in visited:
                alternate_path = list(path)
                alternate_path.append(elements[index])
                DFS_adjacencymatrix_Using_Recursion(begin, dest, alternate_path, visited)        #Recursion
            index+=1

elements={0:'S',1:'a',2:'b',3:'c',4:'d',5:'e',6:'f',7:'G',8:'h',9:'p',10:'q',11:'r'}   #elements mapping to respective values in matrix

Elements_matrix=[[0,0,0,0,1,1,0,0,0,1,0,0],                                               #matrix representation of the directed graph
                 [0,0,0,0,0,0,0,0,0,0,0,0],
                 [0,1,0,0,0,0,0,0,0,0,0,0],
                 [0,1,0,0,0,0,0,0,0,0,0,0],
                 [0,0,1,1,0,1,0,0,0,0,0,0],
                 [0,0,0,0,1,0,0,0,1,0,0,1],
                 [0,0,0,1,0,0,0,1,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,0,0,0,1,1,0],
                 [0,0,0,0,0,0,0,0,0,0,1,0],
                 [0,0,0,0,0,0,0,0,0,0,0,0],
                 [0,0,0,0,0,0,1,0,0,0,0,0]]


print("States Expanded =","->".join(DFS_expanded_states_recursion('S',[]))) ##calling method which returns paths expanded
DFS_adjacencymatrix_Using_Recursion('S','G',['S'],[])                       #calling recursive method which returns DFS shortest path
