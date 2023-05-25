library("eRm")
library("feather")

X <- read.csv("matrix.csv",as.is=TRUE,row.names="EXAMEN")
rasch <- RM(X)
difficulty <- -rasch$betapar
person <- person.parameter(rasch)
ability <- person$theta.table$`Person Parameter`
write.csv(difficulty,"difficulty.csv")
write.csv(ability,"ability.csv")