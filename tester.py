import functions
import matplotlib.pyplot as plt
import glob
import numpy as np

#print(functions.intersect([[0, 0], [1, 1]], [[0, 1], [1, 0]]))
"""for filepath in glob.iglob('img/*.png'):
    print(filepath.split('\\')[-1].split('.')[0].replace('_', '.'))
points = functions.identifyPoints(fileName="cp.png",
                                  pointQuality=0.1,
                                  minDistance=5)
print("Kamiya Ref")
print(functions.kamiyaRefTester(points, 5))"""
scaled_kref = {'og_ref': [((-200, -200), (714.284, 200)), np.array([[ 200., -200.],
       [ 175.,  200.]])]}
print((tuple(map(int, scaled_kref['og_ref'][0][0]))),
                       tuple(map(int, scaled_kref['og_ref'][0][1])))
plt.imshow(functions.identifyPoints(fileName="cp.png", pointQuality=0.1, minDistance=5))
