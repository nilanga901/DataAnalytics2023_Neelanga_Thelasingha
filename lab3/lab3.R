set.seed(12345)
help(par)
# par can be used to set or query graphical parameters.
# Parameters can be set by specifying them as arguments
# to par in tag = value form, or by passing them as a list of tagged values.
par(mar = rep(0.2,4))
data_Matrix <-matrix(rnorm(400), nrow = 40)
image(1:10, 1:40, t(data_Matrix)[,nrow(data_Matrix):1])
help("heatmap") 
help(rep)

par(mar=rep(0.2,4))
heatmap(data_Matrix)
help("rbinom")

set.seed(678910)
for(i in 1:40){
  # flipping a coin and getting the data
  coin_Flip <- rbinom(1, size = 1, prob = 0.5)
  # if the coin is "Heads", add a common pattern to that row,
  if(coin_Flip){
    data_Matrix[i, ] <- data_Matrix[i, ] + rep(c(0,3), each =5)
  }
}

par(mar= rep(0.2, 4))
image(1:10, 1:40, t(data_Matrix)[, nrow(data_Matrix):1])

par(mar=rep(0.2, 4))
heatmap(data_Matrix)

hh<- hclust(dist(data_Matrix))
data_Matrix_Ordered<- data_Matrix[hh$order,]
par(mfrow=c(1,3))

image(t(data_Matrix_Ordered)[,nrow(data_Matrix_Ordered):1])
plot(rowMeans(data_Matrix_Ordered),40:1,,xlab="The ROw Mean",ylab="Row",pch=19)
plot(colMeans(data_Matrix_Ordered),xlab="Column",ylab="Column Mean",pch=19)

data("Titanic")


# Load the required library
library(rpart)

# Fit a decision tree model
titanic_tree <- rpart(Survived ~ ., data = Titanic)

# View the decision tree
print(titanic_tree)


# Load the required library
library(party)

# Fit a conditional inference tree model
titanic_ctree <- ctree(Survived ~ ., data = Titanic)

# View the conditional inference tree
plot(titanic_ctree)

# Load the required library for one-hot encoding
library(caret)

# Create dummy variables for categorical columns
dummies <- dummyVars(" ~ .", data = Titanic, fullRank = TRUE)
titanic_encoded <- predict(dummies, newdata = Titanic)

# Subset the data for clustering (excluding the "Survived" column)
titanic_clustering_data <- titanic_encoded[, -1]

# Perform hierarchical clustering
titanic_hclust <- hclust(dist(titanic_clustering_data))
plot(titanic_hclust)
