import queue                                           ##importing queue

graph={'S':['d','e','p'],
       'a':[ ],
       'b':['a'],
       'c':['a'],
       'd':['b','c','e'],
       'e':['h','r'],                         ###input 2 undirected graph
       'f':['c','G'],
       'G':[ ],
       'h':['p','q'],
       'p':['q'],
       'q':[ ],
       'r':['f'],
}

def bfs_shortpath(graph, begin, dest):              ###called method which returns shortest path
    visitedvertex= set()
    path_q=queue.Queue(maxsize=0)                   ##creating a queue with variable path_q
    path_q.put([begin])
    while path_q:
        path = path_q.get()
        node = path[-1]                            ##assigns node variable with last element of the path
        if node not in visitedvertex:
            for i in graph[node]:
                alternate_path = list(path)
                alternate_path.append(i)
                path_q.put(alternate_path)
            visitedvertex.add(node)
        if node == dest:
            return path

def bfs_connected_component(graph, start):         ##Method which returns States Expanded
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

print("States Expanded = ","->".join(bfs_connected_component(graph,'S')) )     ###calling bfs_connected_component which returns expanded states
print("Path =","->".join(bfs_shortpath(graph, 'S', 'G')) )           ###calling shortest path method(bfs_shortpath(...))with source and destination
