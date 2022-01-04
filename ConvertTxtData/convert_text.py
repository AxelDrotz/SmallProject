import glob
import sklearn.feature_extraction
import sklearn.preprocessing
file_paths = glob.glob('text/*/*.txt')
print(len(file_paths))

def readTextFiles(): 
    file_data = []
    #Open all the text files and extract the text into a list of files
    for files in file_paths:
        with open(files) as f:
            file_data.append(f.read())


    #For exact number of features(Amount of unique tokens/columns), smaller matrix, memory heavy due to storing tokens
    count_vectorizer = sklearn.feature_extraction.text.CountVectorizer()     #Using exact number of features as unique tokens
    tf_matrix = count_vectorizer.fit_transform(file_data)
    #tf_matrix = sklearn.preprocessing.normalize(tf_matrix,axis=1)


    #For a larger number of features(columns), sparse matrix, memory effective, using hash-table
    #vectorizer = sklearn.feature_extraction.text.HashingVectorizer()
    #tf_matrix = vectorizer.fit_transform(file_data)

    return tf_matrix    #Columns as datapoints, rows as dimensions
image_matrix = readTextFiles()
print(image_matrix)
