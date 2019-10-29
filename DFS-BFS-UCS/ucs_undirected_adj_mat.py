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
            if neighbours[k]>0 and pair[k] not in visited_vertex:
                visited_vertex.append(pair[k])
                path.append(pair[k])
            k+=1
    return visited_vertex

def ucs_shortpath(adj_mat, begin, dest,pair):              ###called method which returns shortest path
    visitedvertex= set()
    path_q=queue.PriorityQueue(maxsize=0)                   ##creating a queue with variable path_q
    path_q.put((0,[begin]))
    while path_q:
        temp,path = path_q.get()
        node = path[-1]                            ##assigns node variable with last element of the path
        if node == dest:
            return path,temp
        elif node not in visitedvertex:
            for key, value in pair.items():
                if value == node:
                    current_ele = key
            temp_mat = adj_mat[current_ele]
            count = 0
            while count < len(temp_mat):
                sum=0
                alternate_path = list(path)
                if temp_mat[count]>0:
                    alternate_path.append(pair[count])
                    sum = temp + temp_mat[count]
                path_q.put((sum,alternate_path))
                count+=1
            visitedvertex.add(node)



pair={0:'S',1:'a',2:'b',3:'c',4:'d',5:'e',6:'f',7:'G',8:'h',9:'p',10:'q',11:'r'}  #allocating identites to matrix levels
adj_mat=[[0,0,0,0,3,9,0,0,0,1,0,0],                                               #matrix representation of the undirected graph
         [0,0,0,0,0,0,0,0,0,0,0,0],
         [0,2,0,0,0,0,0,0,0,0,0,0],
         [0,2,0,0,0,0,0,0,0,0,0,0],
         [0,0,1,8,0,2,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,8,0,0,2],
         [0,0,0,0,0,0,0,2,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,0,0,0,4,4,0],
         [1,0,0,0,0,0,0,0,0,0,15,0],
         [0,0,0,0,0,0,0,0,0,0,0,0],
         [0,0,0,0,0,0,2,0,0,0,0,0]]


print("states Expanded =","->".join(mat_to_queue(pair,adj_mat,'S')))              #calling method mat_to_queue which returns states expanded)
#print("Path =","->".join(ucs_shortpath(adj_mat,'S','G',pair)))           #calling method bfs_shortpath which returns shortest path)
ucspath,cost=ucs_shortpath(adj_mat,'S','G',pair)
print("UCS path = ","->".join(ucspath)," Cost= ",cost)