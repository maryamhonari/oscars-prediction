library(lattice)

win_metrics <- read.csv('metrics_win_clf.csv')

colors <- palette(rainbow(nrow(win_metrics))) 

win_table <- data.matrix(win_metrics, rownames.force=NA)

win_table <- win_table[, -1]

barplot(win_table, main="Winner Metrics", xlab="Metric", col=colors, beside=TRUE)

par(xpd=TRUE)
legend(17, 0.67, win_metrics$Algorithm, lty=c(1,1), lwd=c(2.5, 2.5), col=colors)