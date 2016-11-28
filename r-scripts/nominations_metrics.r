library(lattice)

nom_metrics <- read.csv('metrics_nom_clf.csv')

colors <- palette(rainbow(nrow(nom_metrics))) 

nom_table <- data.matrix(nom_metrics, rownames.force=NA)

nom_table <- nom_table[, -1]

barplot(nom_table, main="Nominee Metrics", xlab="Metric", col=colors, beside=TRUE)

par(xpd=TRUE)
legend(11, 1.07, nom_metrics$Algorithm, lty=c(1,1), lwd=c(2.5, 2.5), col=colors)