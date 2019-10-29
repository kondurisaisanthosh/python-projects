def Dfs_shortpath(adj_mat, source, dest,pair):             # method which returns shortest path
    visited_vertex = set()
    stack1=[]
    stack1.append([source])
    while stack1:
        path=stack1.pop(-1)
        node=path[-1]
        if node==dest:
            return path
        elif node not in visited_vertex:
            for key, value in pair.items():
                if value == node:
                    current_ele = key
            temp_mat = adj_mat[current_ele]
            count=len(temp_mat)-1
            while count >= 0:
                alternate_path = list(path)
                if temp_mat[count]==1:
                    alternate_path.append(pair[count])
                    stack1.append(alternate_path)
                count-=1
            visited_vertex.add(node)

def DFS_pathexpanded(Elements_matrix,elements,start,dest):     #method which returns expanded path
    stack1=[start]                                             #initialiizing stack
    path=[]
    while stack1:
        temp=0
        vertex=stack1.pop()                                     #removing last element from the stack
        for key,value in elements.items():
            if value==vertex:
                temp=key
        if vertex in path:
            continue
        path.append(vertex)
        neighbour_mat=Elements_matrix[temp]
        k=len(neighbour_mat)-1
        while k>=0:
            if neighbour_mat[k]==1:
                stack1.append(elements[k])                       #pushing elements onto the stack
            k-=1
    return path


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

print("Paths Expanded =","->".join(DFS_pathexpanded(Elements_matrix,elements,'S','G')))   ##calling method which returns paths expanded
print("Path =","->".join(Dfs_shortpath(Elements_matrix,'S', 'G',elements)))  #calling method which returns DFS shortest path
