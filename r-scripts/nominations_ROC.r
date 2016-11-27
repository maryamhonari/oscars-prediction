library(ROCR)

nominations <- read.csv('Prediction Results - Test-Set Nominations Predictions.csv')

pred <- prediction(nominations$Perceptron, nominations$True.Label)
perf <- performance(pred, measure="tpr", x.measure = "fpr")

plot(perf, col=rainbow(10))