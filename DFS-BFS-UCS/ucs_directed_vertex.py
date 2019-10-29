import queue                                           ##importing queue

graph={'S':{'d':3,'e':9,'p':1},
       'a':{ },
       'b':{'a':2},
       'c':{'a':2},
       'd':{'b':1,'c':8,'e':2},
       'e':{'h':8,'r':2},                         ###input undirected graph
       'f':{'c':3,'G':2},
       'G':{ },
       'h':{'p':4,'q':4},
       'p':{'q':15},
       'q':{ },
       'r':{'f':2},
}

def ucs_shortpath(graph, begin, dest):              ###called method which returns shortest path
    visitedvertex= set()
    path_q=queue.PriorityQueue(maxsize=0)                   ##creating a queue with variable path_q
    path_q.put((0,[begin]))
    while path_q:
        temp,path = path_q.get()
        node = path[-1]                            ##assigns node variable with last element of the path
        if node not in visitedvertex:
            for i in graph[node]:
                sum=0
                alternate_path = list(path)
                sum = temp +graph[node][i]
                alternate_path.append(i)
                path_q.put((sum,alternate_path))
            visitedvertex.add(node)
        if node == dest:
            return path,temp

def ucs_connected_component(graph, start):         ##Method which returns States Expanded
    explored = []
    queue = [start]
    visited= [start]
    while queue:
        node = queue.pop(0)
        explored.append(node)
        neighbours = graph[node]
        for i in neighbours:
            if i not in visited:
                queue.append(i)
                visited.append(i)
    return explored

print("States Expanded =","->".join(ucs_connected_component(graph,'S') ))  #calling bfs_connected_component which returns expanded states
ucspath,cost= ucs_shortpath(graph, 'S', 'G') #calling shortest path method(bfs_shortpath(...))with source and destination
print("UCS short path =","->".join(ucspath),"and cost=",cost)
