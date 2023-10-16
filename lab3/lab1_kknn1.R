
library(kknn)

# Load the iris dataset
data(iris)

m <- dim(iris)[1]
val <- sample(1:m, size = round(m/3), replace = FALSE, prob = rep(1/m, m))

iris.learn <- iris[-val,]
iris.valid <- iris[val,]

iris.kknn <- kknn(Species ~ ., iris.learn, iris.valid, distance = 1, kernel = "triangular")

# Summary of the kknn model
summary(iris.kknn)

fit <- fitted(iris.kknn)

# Create a confusion matrix
table(iris.valid$Species, fit)

# Create a color vector for plotting
pcol <- as.character(as.numeric(iris.valid$Species))

# Fix the typo in the "col" argument
pairs(iris.valid[1:4], pch = pcol, col = c("green3", "red")[as.numeric(iris.valid$Species != fit) + 1])


