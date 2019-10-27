from sklearn.feature_extraction.text import CountVectorizer
from scipy.spatial.distance import cosine
s1 = 'BITS WILP course on data structures'
s2 = 'data mining related course at BITS WILP'

cv = CountVectorizer()
cv.fit([s1,s2])
cvs1 = cv.transform([s1])  ##converts s1 into a vector
cvs2 = cv.transform([s2]) ##converts s2 into a vector

print 1-cosine(cvs1.todense(),cvs2.todense())
## Cosine similarity comments

