t= corona_copy_clean
attach(t)

#changing gender to int
t[gender=="female","gender"]=0
t[gender=="male","gender"]=1

col =c("country","gender","age","visiting Wuhan","from Wuhan")
library(tree)
treefit = tree(death ~ country+age+gender, data=t)
summary(treefit)
plot(treefit)
text(treefit,cex=0.7)

prune1 = prune.tree(treefit,best=3) # Returns best pruned tree
plot(prune1)
text(prune1,cex=0.7)

glm = glm(death~age+gender,data = t)

table(originalData[,]$Survived,originalData[,]$Sex
      
# חלוקת הדאטה לפי טווח גיל ומתו או לא מתו
table(t[which(age<=70 & age>=63),]$death,t[which(age<=70 & age>=63),]$age)

#חלוקת דאטה לפ-י טטוח גיל ומתו או לא סכום
table(t[which(age<=67 & age>=66),"death"])
