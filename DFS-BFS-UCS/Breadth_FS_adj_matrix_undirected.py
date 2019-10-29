import queue                                               # import queue

def mat_to_queue(pair,adj_mat,start):                      # method which returns states expanded
    visited_vertex=[start]
    path=[start]
    while path:
        node=path.pop(0)
        current_ele=0
        for key,value in pair.items():                      # assigns node with respective key
            if value==node:
                current_ele=key
        neighbours=adj_mat[current_ele]
        k=0
        while k<len(neighbours):
            if neighbours[k]==1 and pair[k] not in visited_vertex:
                visited_vertex.append(pair[k])
                path.append(pair[k])
            k+=1
    return visited_vertex

def bfs_shortpath(adj_mat, source, dest,pair):             # method which returns shortest path
    visited_vertex = set()
    path_queue=queue.Queue(maxsize=0)
    path_queue.put([source])
    while(path_queue):
        path=path_queue.get()
        node=path[-1]
        if node==dest:
            return path
        elif node not in visited_vertex:
            for key, value in pair.items():
                if value == node:
                    current_ele = key
            temp_mat = adj_mat[current_ele]
            count = 0
            temp_var=''
            while count<len(temp_mat):
                alternate_path = list(path)
                if temp_mat[count]==1:
                    alternate_path.append(pair[count])
                    path_queue.put(alternate_path)
                count+=1
            visited_vertex.add(temp_var)


pair={0:'S',1:'a',2:'b',3:'c',4:'d',5:'e',6:'f',7:'G',8:'h',9:'p',10:'q',11:'r'}  #allocating identites to matrix levels
adj_mat=[[0,0,0,0,1,1,0,0,0,1,0,0],                                               #matrix representation of the undirected graph
         [0,0,1,1,0,0,0,0,0,0,0,0],
         [0,1,0,0,1,0,0,0,0,0,0,0],
         [0,1,0,0,1,0,1,0,0,0,0,0],
         [1,0,1,1,0,1,0,0,0,0,0,0],
         [1,0,0,0,1,0,0,0,1,0,0,1],
         [0,0,0,1,0,0,0,1,0,0,0,1],
         [0,0,0,0,0,0,1,0,0,0,0,0],
         [0,0,0,0,0,1,0,0,0,1,1,0],
         [1,0,0,0,0,0,0,0,1,0,1,0],
         [0,0,0,0,0,0,0,0,1,1,0,0],
         [0,0,0,0,0,1,1,0,0,0,0,0]]


print("states Expanded =","->".join(mat_to_queue(pair,adj_mat,'S')))              #calling method mat_to_queue which returns states expanded)
print("Path =","->".join(bfs_shortpath(adj_mat,'S','G',pair)))           #calling method bfs_shortpath which returns shortest path)