for row in range(k):
        for column in range(d):
            p = np.random.binomial(1, 1/3)
            if p == 0:
                p = np.random.binomial(1, 0.5)
                if p == 1:
                    R[row][column] = math.sqrt(3)
                else:
                    R[row][column] = - math.sqrt(3)
            else:
                R[row][column] = 0