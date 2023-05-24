# Unsupervised-learning-by-R

# 1. Principle Component Analysis (PCA) vs K-Means Clustering
## 1.1 PCA
a)	Generate a simulated data set with 20 observations in each of three classes ( 60 obs*50 variables)

We have to generate 60 observations divided into 3 classes, where each observation is composed by 50 variables. Firstly, the function matrix() helps us to generate 60*50 normal random numbers with a mean zero and standard deviation 1 and set.seed() is necessary because we want our code to reproduce the same set of random numbers. Then, we can check the dimension of the matrix by dim() function.

```ruby
set.seed(2)
x= matrix(rnorm(20*3*50), ncol=50)
```
![image](https://github.com/kimdta/Unsupervised-learning-by-R/assets/133651115/57808e6a-6103-4e42-9288-152be602cbbe)

The plot shows that the original data is scattered in 2 dimensions. Then, this data are manipulated to belong into 3 different groups by changing the mean of first and second 20 observations. All elements in the first 20 observations are reduced by 5, the third 20 observations are increased by 5 and the second 20 observations remains at normal variables with mean around 0.

```ruby
x[1:20,]= x[1:20,]-5
x[41:60,]= x[41:60,]+5
round(apply(x, 2, mean ), 3)
```
Random normal variables after changing their means are not centered around zero anymore except the second group and the variables are dispersed differently.

b) Performing PCA on 60 observations and plot the first two Principle component score vectors

```ruby
true.labels= c(rep(1,20), rep(2,20), rep(3,20))
plot(x, col=true.labels, xlim=c(min(x), max(x)), ylim=c(min(x), max(x)), main="Original data after changing means", pch=19)  
```
![image](https://github.com/kimdta/Unsupervised-learning-by-R/assets/133651115/12b6501c-8bc4-473c-9190-6960aa675dbb)

Firstly, we label the 3 different groups into different colors and three classes appear separately in the plot. We used to apply PCA on standardized data to have better results using function prcomp(), because there are different variances in our data observations. 

```ruby
pc.out= prcomp(x, scale= TRUE)
summary(pc.out)    #give summary of importance indices of output
```
From summary, we obtain the proportion of variance explained (PVE) and cumulative PVE of all 50 PCs. Cumulative PVE of the first three PCs already explain over 95% of variability of the data; exclusively PC1 could explain 94,48% variability of the data and the other principle components just represent a small amount of variance. Thus, variation of the data was mostly captured by the first principle component.

Use plot function to plot the PVE and cumulative PVE.The elements of pve and cumsum(pve) can be obtained directly from summary(pc.out)$importance

```ruby
par(mfrow=c(1,2))
plot(summary(pc.out)$importance[2,], type="o", main="PVE", col=2, xlab=" ", ylab = " ")
plot(summary(pc.out)$importance[3,],type="o",main="Cumulative PVE", col=3, xlab=" ", ylab = " ")
abline(h=0.95,col=4)
par(mfrow=c(1,1))
```
![image](https://github.com/kimdta/Unsupervised-learning-by-R/assets/133651115/ae634111-6159-496c-96b9-c21e742ba69d)

Looking at the scree plot, we can notice that the first PC explains a substantial amount of variance. Furthermore, there is a remarkable decrease in the variance explained by further PCs and there is not much gained in cumulative pve if we move along the plot further. 

Since the first two principle components have explained pretty large variability of the data (roughly 95%), we plot the first two PCs. On the whole, true.labels will be used to assign a color to each of 60 variables projections based on the group to which it corresponds. It can be seen clearly that variables corresponding to a single group tend to have similar values on the first principle component score vectors, therefore, they lie near each other in low-dimensional space and creates 3 distinct groups with different colors. 

```ruby
plot(pc.out$x[,1:2], col=true.labels, xlab= "PC1", ylab="PC2", main="Principle Component score vectors", pch=19)
```
![image](https://github.com/kimdta/Unsupervised-learning-by-R/assets/133651115/976470b4-2b43-4acd-87c6-b57f2173e527)

## 1.2 K-Means Clustering
a) Perform K-Means Clustering With K=3

We run kmeans() function with K=3 and nstart=20, K-means clustering with nstart large, the algorithm behind will generate better result by using multiple random assignments. Function table() is used to see how matched are our cluster assignment and its true clusters.

```ruby
km3.out=kmeans(x, 3, nstart=20)
table(km3.out$cluster, true.labels)
km3.out$cluster
km3.out$betweenss/km3.out$totss
```
   true.labels
     1  2  3
  1  0 20  0
  2  0  0 20
  3 20  0  0
  
  betweenss/km3.out$totss
  0.9452685
  
One way could be comparing the vector above with the real membership indicator vector too see if the k-means algorithm is working well or not. Consequently, K-means clustering perfectly separated observations into 3 distinct clusters with 20 observations each. Another way is checking the ratio of between sum of square and total sum of square. This result is 94.5%, which means that the distance of units between clusters is explained by their membership to different clusters.

Now, we can sort out clusters based on true.labels to have better visualization as below:

```ruby
jnk=km3.out$cluster
jnk[which(km3.out$cluster==2)]=3
jnk[which(km3.out$cluster==3)]=1
jnk[which(km3.out$cluster==1)]=2
km3.out$cluster=jnk
table(km3.out$cluster, true.labels)  #obs are perfectly clustered
```
   true.labels
     1  2  3
  1 20  0  0
  2  0 20  0
  3  0  0 20
  
``ruby
par(mfrow=c(1,2))  
plot(x, col=(true.labels), main="Original data", xlab=expression(X["1"]), ylab=expression(X["2"]), pch=20, cex=1)
plot(x, col=(km3.out$cluster), main="K-Means with K=3", xlab=expression(X["1"]), ylab=expression(X["2"]), pch=20, cex=1)
par(mfrow=c(1,1))    #obs are perfectly clustered
```
![image](https://github.com/kimdta/Unsupervised-learning-by-R/assets/133651115/866ff6be-d267-402e-b032-c3870a43bdbd)

b) Performing K-means clustering with K=2

We proceed the performance of Kmeans exactly as above with K=2 and check the output to see how observations assigned compare with K=3. 

```ruby
km2.out=kmeans(x, 2, nstart=20)
km2.out$betweenss/km2.out$totss     #ratio B/T is smaller than previous result due to mergence of 2 clusters
table(km2.out$cluster, true.labels)    
```  
  0.7083165
  true.labels
     1  2  3
  1 20 20  0
  2  0  0 20
  
The K-means algorithm has separated 60 observations in two classes of 20 and 40 each. The ratio between sum of square and total sum of square now is 70.8%, which is lower than previous performance with K=3.This happened because now the cluster 1 and 2 has been merged into a cluster with 40 observation, which increased the variability inside the cluster. Therefore, Kmeans with K=2 doesn't really represent true clusters of the data.

![image](https://github.com/kimdta/Unsupervised-learning-by-R/assets/133651115/56d7dda6-735d-4376-85b9-ac1066e5eff0)

c) Performing K-means clusterings K=4

```ruby
km4.out=kmeans(x, 4, nstart=20)
km4.out$betweenss/km4.out$totss 
table(km4.out$cluster, true.labels)
```
  0.9471331  
   true.labels
     1  2  3
  1 11  0  0
  2  0  0 20
  3  0 20  0
  4  9  0  0

When the number of cluster increases, 20 observations in cluster 1 was separated into 2 other clusters with 11 and 9 observations respectively. Then, the ratio B/T is higher because of cluster splitting. In other words, the variance of observation between clusters improved, not because of a more precise assignment clusters, but because the clusters are divided into smaller ones which does not accurately represent the true clusters. By the graphical representation, the first cluster is being divided in two different colors, which in fact are not the true clusters.

![image](https://github.com/kimdta/Unsupervised-learning-by-R/assets/133651115/a7636b4d-db34-4752-92bb-114c3ad84512)

Here, we can find a comparison between Kmeans clustering generated (K=3, K=2 and K=4) and the Original data, where all the data are colored according to their assigned clusters. Overall, Kmeans performance with K=3 expresses precisely the true clusters of the data.

d) Perform K-means on the first 2th PCs

```ruby
km.out=kmeans(pc.out$x[,1:2], 3, nstart=20)
km.out
table(km.out$cluster, true.labels) #all obs are perfectly clustered
```
between_SS / total_SS =  99.5 %
 true.labels
     1  2  3
  1 20  0  0
  2  0  0 20
  3  0 20  0
  
We can observed that from the R output there is a ratio B/T of 99.5%. Kmeans clustering on the first two principle component score vectors can give better results than performing clustering on the full data. In this case, we might view the principle component step as one of denoising (removing noise) the data
Table() function shows that all the observations are perfectly clustered once again, 20 observations for each cluster.

```ruby
par(mfrow=c(1,2))  
plot(pc.out$x[,1:2], col=(true.labels), main="Original data", xlab=expression(X["1"]), ylab=expression(X["2"]), pch=16, cex=1)
plot(pc.out$x[,1:2], col=(km.out$cluster), main="K-Means on 2th PCs", xlab=expression(X["1"]), ylab=expression(X["2"]), pch=16)
```
![image](https://github.com/kimdta/Unsupervised-learning-by-R/assets/133651115/a274dd3c-e802-45d9-a1b4-3f93bc578080)

In fact, we are applying kmeans on the scores of principle components in which our data has been rotated in lower dimension. As we can see over the plot: Observations are assigned separately and perfectly into three clusters.

e) Scaling variables to SD=1 and perform K-means K=3

We use function apply() to transform original variables to variables with a standard deviation equal to 1. Then, we apply Kmeans clustering with K=3 on scaled variables.
The ratio B/T is 94.5%, which is result similar to the output of Kmeans clustering with K=3. 

```ruby
apply(scale(x), 2, sd)
km.scale= kmeans(scale(x), 3, nstart=20)
km.scale$betweenss/km.scale$totss 
table(km.scale$cluster, true.labels)
```
  0.9451569
   true.labels
     1  2  3
  1  0  0 20
  2 20  0  0
  3  0 20  0
We can see that three clusters obtained using K-means on scaled data are somehow identical to three groups obtained by plotting the first two principle components except for the way observations are expressed in 2 dimensions or in new rotated axis.

COMPARISION BETWEEN 2 METHODS

By applying Kmeans clustering and PCA on scaled data, we obtain 3 distinct groups with correctly color assignment. Thus, standardization is necessary when observations are expressed in different measurement units or its standard deviation differ from each other.
Furthermore, we know that observations are 50-dimensional, then Kmeans clustering plot expressed the data in 2 dimensions, which not really represent the data. Meanwhile, PCA performance produce a better interpretation since it plot the first two principle components score vectors regardless of number of features in our coordinates.
In this exercise, we know that there were three clusters because we generated the data. However, for real data, if we do not know the true number of clusters, we have to proceed the Kmeans performance with different number of clusters and look for the one with the most useful or interpretable solution.

```ruby
par(mfrow=c(1,2))  
plot(pc.out$x[,1:2], col=true.labels, xlab= "PC1", ylab="PC2", main= "PC score vectors", pch=18)
plot(x, col=(km.scale$cluster), main="K-Means-scaled data ", xlab=expression(X["1"]), ylab=expression(X["2"]), pch=18, cex=1)
par(mfrow=c(1,1)) 
```
![image](https://github.com/kimdta/Unsupervised-learning-by-R/assets/133651115/5c5e0e85-775f-4a5f-9035-d455831734d5)

Above are the results of performing PCA and Kmeans clustering those are popular unsupervised techniques. The left is projections of the 60 variables onto the first two principle components and the right plots the result of Kmeans clustering on scaled data.

