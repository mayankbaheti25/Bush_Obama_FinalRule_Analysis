###################################################################
# ASSESSMENT : Text mining and insights from US regulatory space  #
# Code by : Mayank Baheti                                         #  
# Date: 13/02/2018                                                #
###################################################################



#Clear the Global Environment
rm(list=ls(all=T))

# Function to Install and Load R Packages
Install_And_Load = function(Required_Packages)
{
  Remaining_Packages = Required_Packages[!(Required_Packages %in% installed.packages()[,"Package"])];
  
  if(length(Remaining_Packages)) 
  {
    install.packages(Remaining_Packages);
  }
  for(package_name in Required_Packages)
  {
    library(package_name,character.only=TRUE,quietly=TRUE);
  }
}


# Specify the list of required packages to be installed and load    
Required_Packages=c("tm","textstem","caret","e1071","rpart","MASS","randomForest","inTrees","usdm","devtools","nnet","doMC","stringr","text2vec","purrrlyr","ggplot2","xgboost","wordcloud")

# Call the Function
Install_And_Load(Required_Packages);
# 
# library(tm) #Used for doing the preprocessing of text data
# library(textstem) #mainly used for stemming and lemmatisation
# library(caret) #for confusion Matrix and tuning the SVM model
# library(MASS) #for Step AIC
# library(e1071) # for SVM and SVD
# library(randomForest) #For Randomforest Model
# library(rpart) #For decision tree
# library(rpart.plot) #for plotting the decision tree to see the rules made
# library(inTrees)# to review the rules made by Random Forest
#library(stringr)
#library(text2vec)
#library(glmnet)
# Library for parallel processing
#library(doMC)
registerDoMC(cores=detectCores())  # Use all available cores

#Create and empty data frame so that after each iteration of for loop the data extracted can be apended

bush = setNames(data.frame(matrix(ncol = 8, nrow = 0)), c("title", "type" , "agency_names" ,    "abstract","document_number" ,"html_url" ,"pdf_url", "publication_date"))

#Tried using the Federal register package in R and api, got only the error after getting few pages from the site as "Error in file(file, "rt") : cannot open the connection:HTTP status was '429 Too Many Requests" which  is to stop the spamming of the server so as a work around following is the code which i have used to extract the data, using the URL and giving a system sleep of 60 seconds so that we can get the required data.

#For loop to iterate over 50 pages of the URL
for(i in 1:50)
{
  #Used paste function so that page # can be specified on each iteration of loop
  tmp = read.csv(paste0('https://www.federalregister.gov/documents/search.csv?conditions%5Bpublication_date%5D%5Bgte%5D=01%2F20%2F2001&conditions%5Bpublication_date%5D%5Blte%5D=01%2F20%2F2009&conditions%5Btype%5D%5B%5D=RULE&page=',i),header = T)
  
  #Append to the existing dataframe so that mega dataframe could be formed
  bush = rbind(bush,tmp)
  rm(tmp)#remove varaible
  Sys.sleep(60)#sleep for 60 secs so that server doesn't consider as spam
}


#In the way we extracted the data for Bush time frame we extract the data for Obama as well
obama = setNames(data.frame(matrix(ncol = 8, nrow = 0)), c("title", "type" , "agency_names" ,    "abstract","document_number" ,"html_url" ,"pdf_url", "publication_date"))


for(i in 1:50)
{
  tmp = read.csv(paste0('https://www.federalregister.gov/documents/search.csv?conditions%5Bpublication_date%5D%5Bgte%5D=01%2F20%2F2009&conditions%5Bpublication_date%5D%5Blte%5D=01%2F20%2F2017&conditions%5Btype%5D%5B%5D=RULE&page=',i),header = T)
  obama = rbind(obama,tmp)
  rm(tmp)
  Sys.sleep(60)
}


#Taking only the unique records , if any duplicate are extracted, while fetching
Bush_df = unique(bush)
obama_df = unique(obama)

#Add a target attribute to both dataframe as the president name.
Bush_df = data.frame(Bush_df,President = "Bush")
obama_df = data.frame(obama_df,President = "Obama")

#Now create a new dataframe which is combination of both bush and obama dataset
final_Dataframe = rbind(Bush_df,obama_df)[,c(1,4,9)]

#combining the title and abstract data in a new column 
final_Dataframe$Rule = paste(final_Dataframe$title,final_Dataframe$abstract,sep = "-")



## Topic extraction of the two Government rule

# Topic Modelling for Bush

#Create the tokens 
Bush_tokens = final_Dataframe$Rule[1:10000] %>% 
  tolower %>% removeWords(stopwords("en"))%>%removeNumbers%>%lemmatize_words%>%removePunctuation%>%word_tokenizer

# creates iterators over input objects in order to create vocabularies
it = itoken(Bush_tokens, ids =final_Dataframe$Rule[1:10000], progressbar = FALSE)
v = create_vocabulary(it) %>% 
  prune_vocabulary(term_count_min = 10, doc_proportion_max = 0.2)
vectorizer = vocab_vectorizer(v)

#create the Document Term Matrix
dtm = create_dtm(it, vectorizer, type = "dgTMatrix")

lda_model_bush = LDA$new(n_topics = 7, doc_topic_prior = 0.1, topic_word_prior = 0.01)
doc_topic_distr = 
  lda_model_bush$fit_transform(x = dtm, n_iter = 1000, 
                               convergence_tol = 0.001, n_check_convergence = 25, 
                               progressbar = FALSE,seed = 1234)


barplot(doc_topic_distr[1, ], xlab = "topic", 
        ylab = "proportion", ylim = c(0, 1), 
        names.arg = 1:ncol(doc_topic_distr))


lda_model_bush$get_top_words(n = 20, topic_number = c(2L, 3L, 7L), lambda = 1)

lda_model_bush$get_top_words(n = 20, topic_number = c(2L, 3L, 7L), lambda = 0.2)

###################Topic Modelling for Obama ##############
Obama_tokens = final_Dataframe$Rule[10001:20000] %>% 
  tolower %>% removeWords(stopwords("en"))%>%removeNumbers%>%lemmatize_words%>%removePunctuation%>%word_tokenizer
it = itoken(Obama_tokens, ids =final_Dataframe$Rule[1:10000], progressbar = FALSE)
v = create_vocabulary(it) %>% 
  prune_vocabulary(term_count_min = 10, doc_proportion_max = 0.2)
vectorizer = vocab_vectorizer(v)
dtm = create_dtm(it, vectorizer, type = "dgTMatrix")

lda_model_Obama = LDA$new(n_topics = 7, doc_topic_prior = 0.1, topic_word_prior = 0.01)
doc_topic_distr = 
  lda_model_Obama$fit_transform(x = dtm, n_iter = 1000, 
                                convergence_tol = 0.001, n_check_convergence = 25, 
                                progressbar = FALSE,seed = 1234)


barplot(doc_topic_distr[1, ], xlab = "topic", 
        ylab = "proportion", ylim = c(0, 1), 
        names.arg = 1:ncol(doc_topic_distr))


lda_model_Obama$get_top_words(n = 20, topic_number = c(1L, 4L, 5L), lambda = 1)

lda_model_Obama$get_top_words(n = 20, topic_number = c(1L, 4L, 5L), lambda = 0.2)




#Now Create an ID column in a new Data Frame and add the Rule Column and The Target feature to it
data_1 = data.frame(id=c(1:nrow(final_Dataframe)),final_Dataframe[,c(4,3)])

#sampling the data into train and test
set.seed(12345)
n  = nrow(data_1)
row_numbers = sample(1:n,size = round(0.7*n),replace = F)
train_Data = data_1[row_numbers,]
test_Data = data_1[-row_numbers,]


##### Vectorization #####
# define preprocessing function and tokenization function
prep_fun = tolower
tok_fun = word_tokenizer

#creates iterators over input objects in order to create vocabularies
it_train = itoken(train_Data$Rule,
                   preprocessor = prep_fun,
                   tokenizer = tok_fun,
                   ids = train_Data$id,
                   progressbar = TRUE)

it_test = itoken(test_Data$Rule,
                  preprocessor = prep_fun,
                  tokenizer = tok_fun,
                  ids = test_Data$id,
                  progressbar = TRUE)

# creating vocabulary and document-term matrix
vocab = create_vocabulary(it_train)
vectorizer = vocab_vectorizer(vocab)
dtm_train = create_dtm(it_train, vectorizer)

# define tf-idf model
tfidf = TfIdf$new()

# fit the model to the train data and transform it with the fitted model
dtm_train_tfidf = fit_transform(dtm_train, tfidf)

# apply pre-trained tf-idf transformation to test data
dtm_test_tfidf  = create_dtm(it_test, vectorizer) %>%  transform(tfidf)

# train the model
t1 = Sys.time()

#Glmnet model : model with Regularization(Lasso Regression)
model_glmnet_classifier = cv.glmnet(x = dtm_train_tfidf,
                               y = train_Data[['President']],
                               family = 'binomial',
                               # L1 penalty
                               alpha = 1,
                               # interested in the area under ROC curve
                               type.measure = "auc",
                               # 5-fold cross-validation
                               nfolds = 5,
                               # high value is less accurate, but has faster training
                               thresh = 1e-3,
                               # again lower number of iterations for faster training
                               maxit = 1e3)

#Time taken to train the model
print(difftime(Sys.time(), t1, units = 'mins'))

plot(model_glmnet_classifier)
#AUC Value
print(paste("max AUC =", round(max(model_glmnet_classifier$cvm), 4)))


#Prediction on Train 
pred_glmnet_train = predict(model_glmnet_classifier, dtm_train_tfidf, type = 'response')[ ,1]
pred_glmnet_train = ifelse(pred_glmnet_train>0.5,"Obama","Bush")# Threshold as 0.5, if Greater than 0.5, its Obama else Bush

#Confusion Matrix
confusionMatrix(train_Data$President, pred_glmnet_train)

# Confusion Matrix and Statistics
# 
# Reference
#                  Prediction Bush Obama
# Bush                        6550   450
# Obama                        878  6122
# 
# Accuracy : 0.9051 


#Predict on test
pred_glmnet_test = predict(model_glmnet_classifier, dtm_test_tfidf, type = 'response')[ ,1]
pred_glmnet_test = ifelse(pred_glmnet_test>0.5,"Obama","Bush")# Threshold as 0.5, if Greater than 0.5, its Obama else Bush

#Confusion Matrix
confusionMatrix(test_Data$President, pred_glmnet_test)

# Confusion Matrix and Statistics
# 
# Reference
#                 Prediction Bush Obama
# Bush                       2663   337
# Obama                       563  2437
# 
# Accuracy : 0.85 
#Best Model achieved

# save the best model for future using
saveRDS(model_glmnet_classifier, 'glmnet_classifier.RDS')
#######################################################



#Other Models and Techniques I have Tried


#take it to separate dataframe
text = data.frame(final_Dataframe$Rule)

#make a corpus of the dataframe made using the 
text_corpus = Corpus(DataframeSource(text))

#specific_transform
toString = content_transformer(function(x, from, to) gsub(from, to, x))

#create a function to to do preprocessing in all the corpus
preprocessing =  function(myCorpus) {
  myCorpus = tm_map(myCorpus, PlainTextDocument) # an intermediate preprocessing step
  
  myCorpus = tm_map(myCorpus, content_transformer(tolower)) # converts all text to lower case
  
  myCorpus = tm_map(myCorpus, removePunctuation) #removes punctuation
  
  myCorpus = tm_map(myCorpus, removeNumbers) #removes numbers
  
  myCorpus = tm_map(myCorpus, removeWords, stopwords('english')) #removes common words like "a", "the" etc
  
  myCorpus = tm_map(myCorpus, removeWords, c("also", "e","will")) #remove_own_stopwords
  
  myCorpus = tm_map(myCorpus, stripWhitespace) #removes extra spaces 
  
  myCorpus = tm_map(myCorpus, toString, "America", "U.S.A")# treat both words as same 
  
  myCorpus = tm_map(myCorpus, toString, "states", "state")
  
  #Stem words in the corpus 
  #Content transformer helps to use the functions outside the TM package
  myCorpus = tm_map(myCorpus, content_transformer(lemmatize_words))  
  
}

#apply this function the preprocessing
text_corpus_processed = preprocessing(text_corpus)

#build a document term matrix
text_dtm = DocumentTermMatrix(text_corpus_processed)



#remove the sparsity of the matrix,
text_dtms = removeSparseTerms(text_dtm, 0.9)


## ------------------------------------------------------------------------
freq = colSums(as.matrix(text_dtms))
freq
table(freq)

## ----freq_terms_5000-----------------------------------------------------
findFreqTerms(text_dtms, lowfreq=5000)


## ----word_count----------------------------------------------------------
freq = sort(colSums(as.matrix(text_dtms)), decreasing=TRUE)
head(freq, 14)
wf   = data.frame(word=names(freq), freq=freq)
rownames(wf)= NULL
head(wf)

## ----plot_freq

subset(wf, freq>100)                                                  %>%
  ggplot(aes(word, freq))                                              +
  geom_bar(stat="identity")                                            +
  theme(axis.text.x=element_text(angle=45, hjust=1))


## ----wordcloud_scale,
set.seed(142)
wordcloud(names(freq), freq, min.freq=1000, scale=c(5, .1), colors=brewer.pal(6, "Dark2"))



#function to remove any document with zero words. 
remove_doc_wo_word = function(dtm) {
  
  rowTotals = apply(text_dtms , 1, sum) #Find the sum of words in each Document
  return(text_dtms[rowTotals> 0, ])           #remove all docs without words
}



#################################################################

#Load Topic models

#Run Latent Dirichlet Allocation (LDA) using Gibbs Sampling
#set burn in
burnin =1000
#set iterations
iter=2000
#thin the spaces between samples
thin = 500
#set random starts at 5
nstart =5
#use random integers as seed 
seed = list(1258,109,122887,149037,2)
# return the highest probability as the result
best =TRUE
#set number of topics 
k =7
#run the LDA model
text_ldaOut7 = LDA(text_dtm,k, method="Gibbs", control= list(nstart=nstart, seed = seed, best=best, burnin = burnin, iter = iter, thin=thin))


#view the top 10 terms for each of the 5 topics, create a matrix and write to csv
terms(text_ldaOut7,10)


#docs to topics
ldaOut.topics = as.matrix(topics(text_ldaOut7))
final_Dataframe$Topic = ldaOut.topics

#see the distribution of obama and bush documents
table(final_Dataframe$President,final_Dataframe$Topic)

#probabilities associated with each topic assignment
topicProbabilities = as.data.frame(text_ldaOut7@gamma)

#Creating a dataframe with document term matrix and topic probablities as the independent feature as Presidents as the target
dtm_tm_df = data.frame(as.matrix(text_dtms),topicProbabilities,President = final_Dataframe$President)

#sampling the data into train and test
set.seed(12345)
n  = nrow(dtm_tm_df)
row_numbers = sample(1:n,size = round(0.7*n),replace = F)
train_Data = dtm_tm_df[row_numbers,]
test_Data = dtm_tm_df[-row_numbers,]

## Classification Model building

#Logistic Regression Model

model_glm = glm(formula = President~.,data = train_Data,family = "binomial")
summary(model_glm)
pred_glm_train = predict(model_glm,train_Data,type = "response")
pred_glm_test = predict(model_glm,test_Data[,-53],type = "response")

#set the threshold value to 0.5 as the class is balanced
pred_glm_train = ifelse(pred_glm_train<0.5,"Obama","Bush")
pred_glm_test = ifelse(pred_glm_test<0.5,"Obama","Bush")
confusionMatrix(test_Data$President,pred_glm_test)

# Confusion Matrix and Statistics
# 
# Reference
# Prediction Bush Obama
# Bush  1032  1968
# Obama 1722  1278
# 
# Accuracy : 0.385 
#worst then random guessing 


#Decision Tree
model_rpart = rpart(President~.,data = train_Data,method = "class")
summary(model_rpart)


pred_rpart_test = predict(model_rpart,test_Data[,-53],type = "class")
#pred_rpart_test = ifelse(pred_rpart_test[,1]>0.5,"Bush","Obama")
confusionMatrix(test_Data$President,pred_rpart_test)

# Confusion Matrix and Statistics
# 
# Reference
#            Prediction Bush Obama
# Bush                  2361   639
# Obama                 2129   871
# 
# Accuracy : 0.5387
#not a good model

#see if there is way to improve the cost complexity
printcp(model_rpart)

#plot the decision tree to see the rules
rpart.plot(model_rpart)


##Random Forest Model

#random forest with 500 tress and 5 as number of features to used to perform bagging
model_rf = randomForest(President~.,data = train_Data,ntree = 500, mtry = 5)

#do the prediction on the unseen dataset(test)
pred_rf_test = predict(model_rf,test_Data[,-53])

#Evaluate the performance of the model
confusionMatrix(test_Data$President,pred_rf_test)

# Confusion Matrix and Statistics
# 
# Reference
#            Prediction Bush Obama
# Bush                  2351   649
# Obama                  815  2185
# 
# Accuracy : 0.756 
#better than decision tree


## Look at variable importance:
att=round(importance(model_rf), 2)
attr=att[order(att,decreasing = T),]

attr=varImp(model_rf, scale=FALSE)

#save the attribute number in a new variable
temp=order(attr,decreasing = T)

#Now subsetting the train and test with only top 30 feature we got from random forest
train_subset = subset(train_Data,select = c(temp[1:30],53))
test_subset = subset(test_Data,select =  c(temp[1:30],53))

# make another randomforest model with only the top 30 important feartures
model_rf_if = randomForest(President~.,data = train_subset,ntree = 500, mtry = 5)

# perform prediction on train and test data set and see the evalution metric
pred_rf_train_if = predict(model_rf_if,train_subset[,-31])
confusionMatrix(train_subset$President,pred_rf_train_if)

pred_rf_test_if = predict(model_rf_if,test_subset[,-31])
confusionMatrix(test_subset$President,pred_rf_test_if)

# Confusion Matrix and Statistics
# 
# Reference
#               Prediction Bush Obama
#  Bush                    2288   712
#  Obama                    847  2153
# 
# Accuracy : 0.7402


#Bayesglm model
model_bayes = train(President ~ ., data = train_subset, method = 'bayesglm')

# Check accuracy on training.
pred_train_bayes = predict(model_bayes, newdata = train_subset[,-31])
confusionMatrix(train_subset$President,pred_train_bayes)

confusionMatrix(test_subset$President,predict(model_bayes, newdata = test_subset[,-31]))

# Confusion Matrix and Statistics
# 
# Reference
#           Prediction Bush Obama
# Bush                 2102   898
# Obama                1478  1522
# 
# Accuracy : 0.604  
#Not a good Model

# Train the classifier
system.time( model_nb = naiveBayes(train_subset, train_subset$President, laplace = 1) )

# Use the NB classifier we built to make predictions on the test set.
system.time( pred_nb_test = predict(model_nb, newdata=test_subset[,-31]) )

confusionMatrix(pred_nb_test,test_subset$President)
#####Another Approach####################
## Perform SVD to extract the hidden features associated with document and words using document term matrix
SVD = svd(text_dtms)

# Extract u matrix which is matrix of Document and its features.
s = diag(SVD[[1]]) # Eigen values
u = SVD[[2]]# Document and it's feature space
v = SVD[[3]]# word and it's feature space

#Make a dataframe for modelling with this SVD feature space, DTM, Topic probability as independent attribute
df_svd = data.frame(as.matrix(u),as.matrix(text_dtms),topicProbabilities,President = final_Dataframe$President)


#sampling the data into train and test
set.seed(12345)
n  = nrow(df_svd)
row_numbers = sample(1:n,size = round(0.7*n),replace = F)
train_svd = df_svd[row_numbers,]
test_svd = df_svd[-row_numbers,]


#Random forest model
model_rf_svd = randomForest(President~.,data = train_svd,ntree = 500)

#perform the prediction on test data
pred_rf_svd_test = predict(model_rf_svd,test_svd[,-98])
confusionMatrix(test_svd$President,pred_rf_svd_test)

# Confusion Matrix and Statistics
# 
# Reference
#             Prediction Bush Obama
# Bush                   2363   637
# Obama                   812  2188
# 
# Accuracy : 0.7585
##Not a better model than the previous random forest model

## Look at variable importance:
att=round(importance(model_rf_svd), 2)
attr=att[order(att,decreasing = T),]
attr=varImp(model_rf_svd, scale=FALSE)

#save the attribute number in a new variable
temp=order(attr,decreasing = T)

#Now subsetting only top 60 feature we got from random forest
train_svd_subset = subset(train_svd,select = c(temp[1:60],98))
test_svd_subset = subset(test_svd,select =  c(temp[1:60],98))

model_rf_svd_if = randomForest(President~.,data = train_svd_subset,ntree = 500, mtry = 7)

pred_rf_svd_train_if = predict(model_rf_svd_if,train_svd_subset[,-61])
confusionMatrix(train_svd_subset$President,pred_rf_svd_train_if)

# Confusion Matrix and Statistics
# 
# Reference
#           Prediction Bush Obama
# Bush                 2386   614
# Obama                 812  2386
#
#Accuracy :  0.7623   


pred_rf_svd_test_if = predict(model_rf_svd_if,test_svd_subset[,-61])
confusionMatrix(test_svd_subset$President,pred_rf_svd_test_if)



treeList = RF2List(model_rf_svd_if)  # transform rf object to an inTrees' format
exec = extractRules(treeList, train_svd_subset[,-61])  # R-executable conditions
exec[1:2,]

ruleMetric = getRuleMetric(exec,train_svd_subset[,-61],train_svd_subset[,61])  # get rule metrics
ruleMetric[1:2,]




#Decision Tree
model_svd_if_rpart = rpart(President~.,data = train_svd_subset,method = "class")
summary(model_rpart)


pred_svd_if_rpart_test = predict(model_svd_if_rpart,test_svd_subset[,-61],type = "class")

confusionMatrix(test_svd_subset$President,pred_svd_if_rpart_test)


#Building an ensemble of all the models
ensemble_train_df = data.frame(pred_glm_train,pred_rf_svd_train_if,pred_rf_train_if,President = train_Data$President)

ensemble_test_df = data.frame(pred_glm_test,pred_rf_svd_test_if,pred_rf_test_if,President = test_Data$President)

#Use a meta learner now as decision tree as its best when we have categorical features

#Decision Tree
model_ens = rpart(President~.,data = ensemble_train_df,method = "class")
summary(model_ens)


pred_ens = predict(model_ens,ensemble_test_df[,-4],type = "class")

confusionMatrix(test_svd_subset$President,pred_svd_if_rpart_test)
# Accuracy : 0.6268 
#Not a good model


#Neural Net

#0 for Obama and 1 for Bush, converting into Numeric in train and test
train_target = ifelse(train_svd_subset$President =="Obama",0,1)
train_nnet = data.frame(train_svd_subset[,-61],President = train_target)

test_target = ifelse(test_svd_subset$President =="Obama",0,1)
test_nnet = data.frame(test_svd_subset[,-61],President = test_target)

# Neural net model

model_nnet = nnet(President~.,data = train_nnet,size=3,hidden = 8, maxit = 1000,decay=10e-4,learningrate = 0.2,trace =T)
pred_nnet_train = predict(model_nnet,train_nnet[-61])
pred_nnet_train = ifelse(pred_nnet_train<0.5,0,1)# 0 for Obama and 1 for Bush
confusionMatrix(train_nnet$President,pred_nnet_train)

# Confusion Matrix and Statistics
# 
# Reference
#               Prediction    0    1
# 0                         4982 2018
# 1                         2242 4758
# 
# Accuracy : 0.6957   

#XGBOOST Model


# fit the model
dtrain = xgb.DMatrix(data = as.matrix(train_nnet[,-61]),
                     label = train_nnet$President)

model_xg = xgboost(data = dtrain, max.depth = 4, 
                   eta = 1, nthread = 2, nround = 100, 
                   objective = "binary:logistic", verbose = 1)
pred_xgb_train = predict(model_xg,as.matrix(train_nnet[,-61]))
pred_xgb_train = ifelse(pred_nnet_train<0.5,0,1)# 0 for Obama and 1 for Bush
confusionMatrix(train_nnet$President,pred_nnet_train)

# Confusion Matrix and Statistics
# 
# Reference
#               Prediction    0    1
# 0                         5685 1315
# 1                         3647 3353
# 
# Accuracy : 0.6456 



# Hence the best model so far formed is using GLMNET model(glm_classifier) giving us an accuracy of 84.9%
################################################################################
