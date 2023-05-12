#importing packages.

library(tidymodels)
library(visdat)
library(tidyr)
library(car)
library(pROC)
library(ggplot2)
library(tidyr)
library(ROCit)
library(stringr)
library(randomForest)


#loading the data(train and test data)
bk_train = read.csv(r"(C:\DATASETS\bank-full_train.csv)",stringsAsFactors = FALSE)

bk_test = read.csv(r"(C:\DATASETS\bank-full_test.csv)",stringsAsFactors = FALSE)


#info of train and test data
head(bk_train)
head(bk_test)
glimpse(bk_train)
glimpse(bk_test)

##checking na values in train and test columns
lapply(bk_train,function(x)sum(is.na(x)))
lapply(bk_test,function(x)sum(is.na(x)))



#converting train and test data into numeric
bk_train$loan = as.numeric(ifelse(bk_train$loan == "yes",1,0))
bk_train$housing = as.numeric(ifelse(bk_train$housing == "yes",1,0))
bk_train$default = as.numeric(ifelse(bk_train$default == "yes",1,0))
bk_train$y=as.numeric(ifelse(bk_train$y == "yes",1,0))

bk_test$loan = as.numeric(ifelse(bk_test$loan == "yes",1,0))
bk_test$housing = as.numeric(ifelse(bk_test$housing == "yes",1,0))
bk_test$default = as.numeric(bk_test$default == "yes")

table(bk_train$y)



#Data Preparation and Creating dummies

dp_pipe=recipe(y~.,data=bk_train) %>%
  update_role(job,marital,education,contact,month,poutcome,new_role="to_dummies") %>% 
  step_rm(has_role("drop_vars")) %>% 
  step_unknown(has_role("to_dummies"),new_level="__missing__") %>% 
  step_other(has_role("to_dummies"),threshold =0.01,other="__other__") %>% 
  step_dummy(has_role("to_dummies")) %>%
  step_impute_median(all_numeric(), -all_outcomes())

dp_pipe=prep(dp_pipe)


## final transformation -> from fit 
train=bake(dp_pipe,new_data=NULL)
test=bake(dp_pipe,new_data=bk_test)

glimpse(train)

lapply(train,function(x)sum(is.na(x)))


#Next we will break our train data into 2 parts in 80:20 ratio. We will build model on one part & check its performance on the other.
set.seed(2)
dim(train)
s=sample(1:nrow(train),0.8*nrow(train))
t1=train[s,] ## you create as model
t2=train[-s,] ## you validate here in this dataset
head(t1)


#Model Building
####We will use train for logistic regression model building and use t2 to test the performance of the model thus built.


####Lets build logistic regression model on train dataset.
fit = lm(y~. -ID -default -loan -housing-marital_X__other__-education_X__other__ -contact_X__other__ -poutcome_X__other__,data = t1)
vif(fit)
alias(fit)
#logistic Regression
logfit = glm(y~.-ID-marital_X__other__-education_X__other__ -contact_X__other__ -poutcome_X__other__-poutcome_unknown-age-poutcome_other-default-previous-month_aug-month_feb -month_jun -job_entrepreneur -job_management -job_retired -job_self.employed -job_technician -job_student -job_services -job_unemployed-job_X__other__-marital_single -education_unknown -contact_telephone,data=t1 ,family='binomial')

#remove linear colinearity we used alias
alias(logfit)
#In order to take care of multi collinearity,we remove variables whose VIF>5,as follows:
sort(vif(logfit),decreasing=TRUE)[1:3]
alias(logfit)
logfit=step(logfit)
summary(logfit)

#lets make predict score
val.score = predict(logfit,newdata=t2,type='response')


logfitfinal = glm(y~.-marital_X__other__-education_X__other__ -contact_X__other__ -poutcome_X__other__-poutcome_unknown-age-poutcome_other-default-previous-month_aug-month_feb -month_jun -job_entrepreneur -job_management -job_retired -job_self.employed -job_technician -job_student -job_services -job_unemployed-job_X__other__-marital_single -education_unknown -contact_telephone-month_sep-month_X__other__,data=train,family='binomial')
alias(logfitfinal)
sort(vif(logfitfinal),decreasing=TRUE)[1:3]
summary(logfitfinal)



train$score = predict(logfitfinal,newdata=train,type='response')


###Step 4. Finding Cutoff value and Perfomance measurements of the model.
cutoff_data=data.frame(cutoff=0,TP=0,FP=0,FN=0,TN=0)
cutoffs=seq(0,1,length=100)


for (i in cutoffs){
  predicted=as.numeric(train$score>i)
  
  TP=sum(predicted==1 & train$y==1)
  FP=sum(predicted==1 & train$y==0)
  FN=sum(predicted==0 & train$y==1)
  TN=sum(predicted==0 & train$y==0)
  cutoff_data=rbind(cutoff_data,c(i,TP,FP,FN,TN))
}
## lets remove the dummy data cotaining top row in data frame cutoff_data
cutoff_data=cutoff_data[-1,]
#we now have 100 obs in df cutoff_data
```
####lets calculate the performance measures:sensitivity,specificity,accuracy, KS and precision.

cutoff_data=cutoff_data %>%
  mutate(P=FN+TP,N=TN+FP, #total positives and negatives
         Sn=TP/P, #sensitivity
         Sp=TN/N, #specificity
         KS=abs((TP/P)-(FP/N)),
         Accuracy=(TP+TN)/(P+N),
         Lift=(TP/P)/((TP+FP)/(P+N)),
         Precision=TP/(TP+FP),
         Recall=TP/P
  ) %>% 
  select(-P,-N)

#lets view cutoff dataset:
 
View(cutoff_data)
```
####Lets find cutoff value based on ks MAXIMUM.

KS_cutoff=cutoff_data$cutoff[which.max(cutoff_data$KS)]
KS_cutoff

#lets predict test score
test$score=predict(logfitfinal,newdata =test,type = "response")
test$left=as.numeric(test$score>KS_cutoff)#if score is greater dan cutoff then true(1) else false(0)
table(test$left)

test$leftfinal=factor(test$left,levels = c(0,1),labels=c("no","yes"))
table(test$leftfinal)

P=FN+TP 
N=TN+FP
Accuracy=(TP+TN)/(P+N)
(770+5888)/7912

#error will be
error=1- Accuracy
###Step 6:Creating confusion matrix and find how good our model is (by predicting on t2 dataset)
t2$score=predict(logfitfinal,newdata =t2,type = "response")

table(t2$y,as.numeric(t2$score>KS_cutoff))
table(test_25$y)

####Lets plot the ROC curve:
library(pROC)

roccurve=roc(t2$y,t2$score) #real outcome and predicted score is plotted
plot(roccurve)


#Thus area under the ROC curve is:
auc(roccurve)

