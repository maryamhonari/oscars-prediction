library(ROCR)

nominations <- read.csv('Prediction Results - Test-Set Nominations Predictions.csv')

plot.new()
plot('False Positive Rate', 'True Positive Rate', main = "Nominations ROC", xlim = c(0, 1), ylim = c(0, 1))

colors <- palette(rainbow(ncol(nominations) - 1))
counter <- 1

for(clf in names(nominations)){
  if(clf != "True.Label") {
    pred <- prediction(nominations[[clf]], nominations$True.Label)
    perf <- performance(pred, measure = "tpr", x.measure = "fpr")
    plot(perf, add = TRUE, col=colors[counter])
    counter <- counter + 1
  }
}
legend(0.725, 0.3, c("Perceptron", "SVM", "AdaBoost"), lty = c(1,1), lwd = c(2.5, 2.5), col = colors)

