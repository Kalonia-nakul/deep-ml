def k_means_clustering(points: list[tuple[float, ...]], k: int, initial_centroids: list[tuple[float, ...]], max_iterations: int) -> list[tuple[float, ...]]:
	# Your code here
	import numpy as np
    import numpy as np
 
    centroids = [list(c) for c in initial_centroids]
 
    def calculate_distance():
        y = [[] for _ in range(len(centroids))]  # fresh list every call, no leftover from last round
 
        for i in range(len(points)):
            temp = []
            for j in range(len(centroids)):
                dis = 0
                for x in range(len(centroids[j])):
                    dis += (centroids[j][x] - points[i][x]) ** 2
                dis = np.sqrt(dis)
                temp.append(dis)
 
            min_idx = 0
            for j in range(len(temp)):
                if temp[j] < temp[min_idx]:
                    min_idx = j
 
            y[min_idx].append(points[i])
 
        return y
 
    for _ in range(max_iterations):
        y = calculate_distance()  # <-- actually call it, and get fresh assignments each round
 
        # mean for centroids
        for i in range(len(y)):
            n = len(y[i])
            if n == 0:
                continue  # keep old centroid if no points assigned this round
 
            n_dims = len(centroids[i])  # <-- loop over dimensions, not number of centroids
            for j in range(n_dims):
                mean = 0
                for x in range(n):
                    mean += y[i][x][j]
                centroids[i][j] = mean / n
 
    return [tuple(c) for c in centroids]
 