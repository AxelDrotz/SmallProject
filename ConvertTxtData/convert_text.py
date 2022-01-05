import glob
import sklearn.feature_extraction
import sklearn.preprocessing
import numpy as np
paths = glob.glob('text/*/*.txt')

print(len(paths))

data = []
#Open all the text files and extract the text into a list of files
for path in paths:
    with open(path) as file:
        data.append(file.read())

count_vectorizer = sklearn.feature_extraction.text.CountVectorizer()
tf_matrix = count_vectorizer.fit_transform(data)
tf_matrix = sklearn.preprocessing.normalize(tf_matrix,axis=1)

a = tf_matrix.transpose()
matrix = a.toarray()
print(type(matrix))
matrix2 = matrix[0:5000]
matrix = matrix2
print(matrix2.shape)
for i in range(len(matrix)):
    temp_sum = np.linalg.norm(matrix[i])
    matrix[i] = matrix[i]/temp_sum
print(matrix.shape)
np.save('textdata.npy', matrix)
