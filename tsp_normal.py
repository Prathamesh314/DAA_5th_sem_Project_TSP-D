arr = [
    [0, 10, 15, 20],
    [5, 0, 9, 10],
    [6, 13, 0, 12],
    [8, 8 ,9 , 0],
]

def tsp(i, start, s):
    if len(s) == 0:
        return [start], arr[i][start]
    ans = 1e9
    path = []
    for j in s:
        s.remove(j)
        rpath, cost = tsp(j, start, s)
        s.add(j)
        cost += arr[i][j]
        rpath.append(j)
        if cost < ans:
            ans = cost
            path = rpath
    return path, ans


s = set([i for i in range(4)])
for i in range(4):
    s.remove(i)
    path, ans = tsp(i,i,s)
    s.add(i)
    path.append(i)
    print(f"Starting vertex: {i+1}")
    print(f"Path: {path}")
    print(f"Cost: {ans}")
    print()
