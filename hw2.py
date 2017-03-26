# import
from __future__ import division
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances, pairwise_distances
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import scipy


def countPositive(matrix, result):
    counterTruePositive = 0;
    counterFalsePositive = 0;
    for x in matrix:
        if str(x) in result:
            counterTruePositive += 1
        else:
            counterFalsePositive += 1
    return counterTruePositive, counterFalsePositive


def countNegative(matrix, result):
    counterFalseNegative = 0;
    counterTrueNegative = 1400 - 10 - len(result) - 10
    for x in result:
        if str(x) not in matrix:
            counterFalseNegative += 1
    return counterFalseNegative, counterTrueNegative


def calculateStatistics(positiveTupple, negativeTupple):
    tp = positiveTupple[0]
    fp = positiveTupple[1]
    tn = negativeTupple[1]
    fn = negativeTupple[0]

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_measure = 0
    if precision > 0 and recall > 0:
        f_measure = 2 * ((precision * recall) / (precision + recall))

    return (precision, recall, f_measure)


def getAveragesByStatistic(inputArray):
    averagePrecision = 0
    averageRecall = 0
    average_measure = 0
    for item in inputArray:
        averagePrecision += item[0]
        averageRecall += item[1]
        average_measure += item[2]
    averagePrecision = averagePrecision / len(inputArray)
    averageRecall = averageRecall / len(inputArray)
    average_measure = average_measure / len(inputArray)

    return ("Average Precision "+str(averagePrecision), "Average Recall "+ str(averageRecall), "Average measure "+str(average_measure))


def countPositiveAndNegative(topRelevant, result):
    positiveTupple = countPositive(topRelevant, result)
    negativeTupple = countNegative(topRelevant, result)
    return (positiveTupple, negativeTupple)


corpus = []

resultCosineTfidf = []
resultCosineBinary = []
resultCosineTfi = []

resultEuclideanTfidf = []
resultEuclideanBinary = []
resultEuclideanTfi = []

for d in range(1400):
    f = open("./d/" + str(d + 1) + ".txt")
    corpus.append(f.read())
for q in range(5):
    if (q > 0):
        corpus.pop()
    f = open("./q/" + str(q + 1) + ".txt")
    corpus.append(f.read())
    tfidf_vectorizer = TfidfVectorizer()

    f = open("./r/" + str(q + 1) + ".txt")
    result = [s.strip() for s in f.read().splitlines()]
    tfidf_matrix = tfidf_vectorizer.fit_transform(corpus)

    tfidf_vectorizer.use_idf = False
    tf_matrix = tfidf_vectorizer.fit_transform(corpus)

    tfidf_vectorizer.binary = True
    tfidf_vectorizer.use_idf = False
    tfidf_vectorizer.norm = False
    binary_matrix = tfidf_vectorizer.fit_transform(corpus)

    sim = np.array(cosine_similarity(tfidf_matrix[len(corpus) - 1], tfidf_matrix[0:(len(corpus) - 1)])[0])
    topRelevant = sim.argsort()[-10:][::-1] + 1
    calculateValues = countPositiveAndNegative(topRelevant, result)
    resultCosineTfidf.append(calculateStatistics(calculateValues[0], calculateValues[1]))

    sim = np.array(cosine_similarity(tf_matrix[len(corpus) - 1], tf_matrix[0:(len(corpus) - 1)])[0])
    topRelevant = sim.argsort()[-10:][::-1] + 1
    calculateValues = countPositiveAndNegative(topRelevant, result)
    resultCosineTfi.append(calculateStatistics(calculateValues[0], calculateValues[1]))

    sim = np.array(cosine_similarity(binary_matrix[len(corpus) - 1], binary_matrix[0:(len(corpus) - 1)])[0])
    topRelevant = sim.argsort()[-10:][::-1] + 1
    calculateValues = countPositiveAndNegative(topRelevant, result)
    resultCosineBinary.append(calculateStatistics(calculateValues[0], calculateValues[1]))

    sim = np.array(euclidean_distances(tfidf_matrix[len(corpus) - 1], tfidf_matrix[0:(len(corpus) - 1)])[0])
    topRelevant = sim.argsort()[::-1][-10:][::-1] + 1
    calculateValues = countPositiveAndNegative(topRelevant, result)
    resultEuclideanTfidf.append(calculateStatistics(calculateValues[0], calculateValues[1]))

    sim = np.array(euclidean_distances(tf_matrix[len(corpus) - 1], tf_matrix[0:(len(corpus) - 1)])[0])
    topRelevant = sim.argsort()[::-1][-10:][::-1] + 1
    calculateValues = countPositiveAndNegative(topRelevant, result)
    resultEuclideanTfi.append(calculateStatistics(calculateValues[0], calculateValues[1]))

    sim = np.array(euclidean_distances(binary_matrix[len(corpus) - 1], binary_matrix[0:(len(corpus) - 1)])[0])
    topRelevant = sim.argsort()[::-1][-10:][::-1] + 1
    calculateValues = countPositiveAndNegative(topRelevant, result)
    resultEuclideanBinary.append(calculateStatistics(calculateValues[0], calculateValues[1]))

print("TDIF statistics with Cosine "+ str(getAveragesByStatistic(resultCosineTfidf)))
print("Binary statistics with Cosine "+ str(getAveragesByStatistic(resultCosineBinary)))
print("TFI statistics with Cosine "+ str(getAveragesByStatistic(resultCosineTfi)))

print("TDIF statistics with Euclidean "+str(getAveragesByStatistic(resultEuclideanTfidf)))
print("Binary statistics with Euclidean "+str(getAveragesByStatistic(resultEuclideanBinary)))
print("TFI statistics with Euclidean "+str(getAveragesByStatistic(resultEuclideanTfi)))
