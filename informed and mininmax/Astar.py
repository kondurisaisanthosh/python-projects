import queue                                           ##importing queue

graph={'S':{'d':3,'e':9,'p':1,'heur':0},
       'a':{'b':2,'c':2,'heur':5},
       'b':{'a':2,'d':1,'heur':7},
       'c':{'a':2,'d':8,'f':3,'heur':4},
       'd':{'b':1,'c':8,'e':2,'S':3,'heur':7},
       'e':{'d':2,'S':9,'h':8,'r':2,'heur':5},                         ###input undirected graph
       'f':{'c':3,'G':2,'r':2,'heur':2},
       'G':{'f':2,'heur':0},
       'h':{'p':4,'q':4,'e':8,'heur':11},
       'p':{'S':1,'h':4,'q':15,'heur':14},
       'q':{'p':15,'h':4,'heur':12},
       'r':{'e':2,'f':2,'heur':3},
}

def aStar(graph, begin, dest):              ###called method which returns shortest path
    visitedvertex= set()
    expandednode=[]
    path_q=queue.PriorityQueue(maxsize=0)                   ##creating a queue with variable path_q
    path_q.put((0,[begin]))
    while path_q:
        cost,path = path_q.get()
        node = path[-1]                            ##assigns node variable with last element of the path
        if node not in visitedvertex:
            expandednode.append(node);
            cost=cost-graph[node]['heur']
            for i in graph[node]:
                if(i!='heur' and i not in visitedvertex):
                    acost =cost+ graph[i]['heur']+graph[node][i]
                    alternate_path = list(path)
                    alternate_path.append(i)
                    path_q.put((acost, alternate_path))
            visitedvertex.add(node)
        if node == dest:
            return path,expandednode



astar,states= aStar(graph, 'S', 'G')
print("A* expanded nodes =","->".join(states))
print("A* short path =","->".join(astar))