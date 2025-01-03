from collections import deque

class GraphTraversal:
    def __init__(self, graph):
        self.graph = graph

    def bfs(self, start_node):
        visited = set()
        queue = deque([start_node])
        traversal_order = []

        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                traversal_order.append(node)

                for neighbor in self.graph[node]:
                    if neighbor not in visited:
                        queue.append(neighbor)

        return traversal_order

    def dfs_iterative(self, start_node):
        visited = set()
        stack = [start_node]
        traversal_order = []

        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                traversal_order.append(node)

                for neighbor in reversed(self.graph[node]):
                    if neighbor not in visited:
                        stack.append(neighbor)

        return traversal_order

    def depth_limited_dfs(self, node, depth, visited, traversal_order):
        if depth == 0:
            return False

        visited.add(node)
        traversal_order.append(node)

        if depth == 1:
            return True

        for neighbor in self.graph[node]:
            if neighbor not in visited:
                if self.depth_limited_dfs(neighbor, depth - 1, visited, traversal_order):
                    return True
        return False

    def iddfs(self, start_node, max_depth):
        for depth in range(1, max_depth + 1):
            visited = set()
            traversal_order = []
            print(f"Depth: {depth}")
            if self.depth_limited_dfs(start_node, depth, visited, traversal_order):
                return traversal_order
            print(f"Traversal Order at Depth {depth}: {traversal_order}")
        return None

graph = {
    'A': ['B', 'C'],
    'B': ['A', 'D', 'E'],
    'C': ['A', 'F'],
    'D': ['B'],
    'E': ['B', 'F'],
    'F': ['C', 'E']
}

traversal = GraphTraversal(graph)

print("BFS:", traversal.bfs('A'))
print("DFS :", traversal.dfs_iterative('A'))
traversal.iddfs('A', 3)

