library(ROCR)

nominations <- read.csv('pred_nom.csv')

plot.new()
plot('False Positive Rate', 'True Positive Rate', main = "Nominations ROC", xlim = c(0, 1), ylim = c(0, 1))

colors <- palette(rainbow(ncol(nominations) - 1))
colors <- replace(colors, colors=="yellow", "black")

counter <- 1

for(clf in names(nominations)){
  if(clf != "True.Label") {
    pred <- prediction(nominations[[clf]], nominations$True.Label)
    perf <- performance(pred, measure = "tpr", x.measure = "fpr")
    plot(perf, add = TRUE, col=colors[counter], lwd=2)
    counter <- counter + 1
  }
}
legend(0.803, 0.33, names(nominations)[-1], lty = c(1,1), lwd = c(2.5, 2.5), col = colors)

