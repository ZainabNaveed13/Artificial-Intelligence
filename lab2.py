class GraphNode:
    def __init__(self, vertex=0, next_node=None):
        self.vertex = vertex
        self.next = next_node
class Graph:

    def __init__(self):
        self.MAX = 10
        self.headnodes = [None] * self.MAX
        self.n = 0

        self.visited = [False] * self.MAX

    def initialize_visited(self):
        self.visited = [False] * self.MAX


    def addVertex(self, vertex):
        new_vertex = GraphNode(vertex)
        if self.n >= self.MAX:
            return "Graph is Full"
        for node in self.headnodes:
            if node and node.vertex == vertex:
                return "Vertex Already Presnt"
        self.headnodes[self.n] = new_vertex
        self.n += 1
        return "Vertex Added"

    def removeVertex(self, vertex):
        if not self.vertexExists(vertex):
            return 'Vertex Not Exist'

        index = self.get_vertex_index(vertex)

        # Remove all edges pointing to this vertex
        for i in range(self.n):
            if self.headnodes[i] is not None and self.headnodes[i].vertex != vertex:
                self.removeEdge(self.headnodes[i].vertex, vertex)

        # Remove all edges from this vertex to others
        curr = self.headnodes[index]
        while curr and curr.next:
            self.removeEdge(vertex, curr.next.vertex)

        # Remove the vertex itself
        self.headnodes[index] = None
        self.n -= 1

    def addEdge(self, vertex1, vertex2):
        if not self.vertexExists(vertex1):
            self.addVertex(vertex1)
        if not self.vertexExists(vertex2):
            self.addVertex(vertex2)

        index1 = self.get_vertex_index(vertex1)
        new_node = GraphNode(vertex2, self.headnodes[index1].next)
        self.headnodes[index1].next = new_node
        return "Edge Added"

    def get_vertex_index(self, vertex):
        for i in  range(self.n):
            if self.headnodes[i] and self.headnodes[i].vertex == vertex:
                return i
                break

    def removeEdge(self, vertex1, vertex2):
        if self.vertexExists(vertex1) and self.vertexExists(vertex2):
            index1 = self.get_vertex_index(vertex1)
            node = self.headnodes[index1].next
            prev = None
            while node:
                if node.vertex == vertex2:
                    if prev:
                        prev.next = node.next
                    else:
                        self.headnodes[index1].next = node.next
                prev = node
                node = node.next


    def vertexExists(self, vertex):
        return self.headnodes[vertex] is not None

    def printGraph(self):
        for i in range(self.MAX):
            if self.headnodes[i] is not None:
                print(f"Vertex {i}:", end="")
                curr = self.headnodes[i].next
                while curr is not None:
                    print(f" {curr.vertex}", end=", ")
                    curr = curr.next
                print()

    def dfs(self, vertex):
        if not self.vertexExists(vertex):
            return "Vertex Does not Exist"
        else:
            self.visited[vertex] = True
            index = self.get_vertex_index(vertex)
            print(self.headnodes[index].vertex, end=" ")
            curr = self.headnodes[index].next
            while curr:
                curr_index = self.get_vertex_index(curr.vertex)
                if curr_index is not None and not self.visited[curr.vertex]:
                    self.dfs(curr.vertex)
                curr = curr.next

    def bfs(self, vertex):
        if self.vertexExists(vertex):
            self.initialize_visited()
            queue = []
            queue.append(vertex)
            self.visited[vertex] = True
            while queue:
                node = queue.pop(0)
                print(node, end = ' ')

                curr = self.headnodes[node].next
                while curr:
                    #print(curr.vertex, end = ' ')
                    if not self.visited[curr.vertex]:

                        queue.append(curr.vertex)
                        self.visited[curr.vertex] = True
                    curr = curr.next


g = Graph()
# Add vertices
for i in range(6):
    g.addVertex(i)
# Add edges
g.addEdge(5, 2)
g.addEdge(5, 0)
g.addEdge(4, 0)
g.addEdge(4, 1)
g.addEdge(2, 3)
g.addEdge(3, 1)



#g.removeVertex(5)
# g.removeEdge(0, 1)
# Print the graph
g.printGraph()
# Perform DFS and BFS traversals
    #print("DFS starting from vertex 0:")
    #g.dfs(0)
#g.initialize_visited()
print()
print("BFS starting from vertex 0:")
g.bfs(2)
