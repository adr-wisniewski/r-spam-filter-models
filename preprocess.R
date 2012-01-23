# setup console
options(width=1000)

#libraries
library("Snowball") # stemmer
library("e1071")	# bayes
library("knnflex")	# knn
library("kernlab")	# svm
library("lsa")		# cosine distance

# preprocessing options
preprocess.options.stemming <- TRUE
preprocess.options.minimumTermLength <- 3
preprocess.options.maximumTermLength <- 50
preprocess.options.minimumTermDocuments <- 10
preprocess.options.stopwords <- SnowballStemmer(c("i", "a", "about", "an", "are", "as", "at", "be", "by", "com", "for", 
		"from", "how", "in", "is", "it", "of", "on", "or", "that", "the ", "this", 
		"to", "was", "what", "when", "where", "who", "will", "with", "the", "www"))

preprocess.readDirectories<-function(dirlist) {
	
	result <- list()
	result$rawDocuments <- c()
	result$classes <- c()
	
	for(dirname in names(dirlist)) {
		documentclass <- dirlist[[dirname]]
		
		for(filename in list.files(dirname)){
			path<-paste(dirname, filename, sep="")
			fileHandle<-file(path, "rt")
			conent<-readChar(fileHandle, 1000000)
			result$rawDocuments<-append(result$rawDocuments, conent)
			result$classes<-append(result$classes, documentclass)
			close(fileHandle)
		}
	}
	
	return(result)
}

preprocess.extractTerms<-function(documents) {
	result <- documents
	# extract email content
	result <- sub("^.*?\n\n", "", result)
	# remove all digits and special characters
	result <- gsub("[^[:alpha:][:blank:]]", " ", result)
	# remove leading and trailing blanks
	result <- sub("^[[:blank:]]*", "", result)
	result <- sub("[[:blank:]]*$", "", result)
	# make lowercase
	result <- tolower(result)
	# split into terms
	result <- strsplit(result, "[[:blank:]]+")
	
	if(preprocess.options.stemming) {
		for(i in 1:length(result)) {
			result[[i]] <- SnowballStemmer(result[[i]])
			result[[i]] <- result[[i]][result[[i]]!=""]
		}
	}
	
	return(result)
}

preprocess.buildDictionary<-function(documents) {
	
	dictionary <- list()

	documentCount <- length(documents)
	for(di in 1:documentCount) {
		document <- documents[[di]]
		print(c("processing document", di, " of ", documentCount, di/documentCount))
		
		documentTerms = list()
		
		for(term in document) {
			documentTerms[[term]] <- ifelse(is.null(documentTerms[[term]]), 0, documentTerms[[term]]) + 1
		}
		
		for(term in names(documentTerms)) {
			dictionary[[term]] <- ifelse(is.null(dictionary[[term]]), 0, dictionary[[term]]) + 1
		}
	}
	
	return(dictionary)
}

preprocess.filterDictionary<-function(dictionary) {
	result <- dictionary
	
	initianlLength <- length(result)
	print(c("Initial dictionary length", initianlLength))
	
	# filter all stopwords
	for(word in preprocess.options.stopwords) {
		result[[word]] <- NULL
	}
	
	# filter all words of too small length or frequency
	for(word in names(dictionary)) {
		if(nchar(word) < preprocess.options.minimumTermLength
			|| nchar(word) > preprocess.options.maximumTermLength
			|| dictionary[[word]] < preprocess.options.minimumTermDocuments) {
			result[[word]] <- NULL
		}
	}
	
	resultLength <- length(result)
	print(c("Dictionary length after filtering", resultLength, "removed", initianlLength - resultLength))
	
	if(length(result) == 0) {
		print("WARNING: DICTIONARY IS EMPTY AFTER FILTERING!")
	}
	
	return(result)
}

preprocess.buildData<-function(documents, classes, dictionary) {
	# create matrix for data
	rows <- length(documents)
	columns <- 1 + length(dictionary)
	result <- matrix(c(0), rows, columns)
	colnames(result) <- c(".", names(dictionary))
	result <- as.data.frame(result)
	
	# fill rows
	documentCount <- length(documents)
	for(di in 1:documentCount) {
		print(c("vectorizing document", di, " of ", documentCount, di/documentCount))
		
		document <- documents[[di]]
		result[[1]][di] <- classes[di]
		
		for(term in document) {
			count <- result[[term]]
			if(!is.null(count)) {
				result[[term]][di] <- count[di] + 1
			}
		}
	}
	
	emptyRows <- c()
	if(columns > 1) {
		for(ri in 1:documentCount) {
			print(c("checking document", ri, " of ", documentCount, ri/documentCount))
			record <- result[ri,-1]
			if(all(record == 0)) {
				emptyRows <- c(emptyRows, ri)
			}
		}
	}

	if(!is.null(emptyRows)) {
		print(c("WARNING: REMOVING EMPTY DOCUMENTS FROM SET", emptyRows))
		result <- result[-emptyRows,]
	}
	
	colnames(result) <- paste("word.", colnames(result), sep="")
	colnames(result)[1] <- "spam"

	# some libs silently require this
	result$spam <- factor(result$spam)
	
	return(result)
}

preprocess.do <- function() {
	

	print("[preprocess] [1/6] reading directories")
	problem <- preprocess.readDirectories(list("./data/test_ham/" = 0, "./data/test_spam/" = 1))
	#problem <- preprocess.readDirectories(list("./data/1_ham/" = 0, "./data/1_spam/" = 1))

	print("[preprocess] [2/6] extracting terms")
	problem$documents <- preprocess.extractTerms(problem$rawDocuments)
	
	print("[preprocess] [3/6] building dictionary")
	problem$dictionary <- preprocess.buildDictionary(problem$documents)
	
	print("[preprocess] [4/6] filtering dictionary")
	problem$dictionary <- preprocess.filterDictionary(problem$dictionary)
	
	print("[preprocess] [5/6] building dataset")
	problem$dataset <- preprocess.buildData(problem$documents, problem$classes, problem$dictionary);
	
	print("[preprocess] [6/6] finished")
	return(problem)
}

transform.toBinary<-function(data) {
	result <- data
	result[result > 1] <- 1
	return(result)
}

transform.toTfIdf<-function(data, dictionary) {
	result <- data
	documentCount <- nrow(data)
	
	# calculate idf
	idf <- c()
	for(termDocumentCount in dictionary) {
		idf <- c(idf, log(documentCount/termDocumentCount))
	}
	
	result[,-1] <- result[,-1] * idf
	return(result)
}

transform.randomize<-function(data) {
	result <- data
	result = result[sample(1:nrow(result), nrow(result)),]
	return(result)
}

spamfilter.opt.nb.threshold <- 0.001

spamfilter.opt.knn.k <- 3
spamfilter.opt.knn.dist <-"euclidean" # "manhattan" # "cosine"

spamfilter.opt.svm.type <- "C-svc"
spamfilter.opt.svm.kernel <- "rbfdot"
spamfilter.opt.svm.cost <- 1

spamfilter.opt.folds <- 5

spamfilter.crossValidate <- function(data, modelFunction, folds) {
	testDataPerFold <- nrow(data) / folds
	allFp <- c(0)
	allFn <- c(0)
	
	print("crossvalidation start")
	
	for(fold in 1:folds) {
		
		testIndices <- ((fold-1)*testDataPerFold):(fold*testDataPerFold)
		dataIndices <- (1:nrow(data))[-testIndices]
		classes <- modelFunction(data, dataIndices, testIndices)
		classes <- as.numeric(as.vector(classes))
		
		fp <- sum(classes == 1 & data[testIndices,1] == 0)
		fn <- sum(classes == 0 & data[testIndices,1] == 1)
		err <- sum(classes != data[testIndices,1])
		
		print(c("fold:", fold, "false positives:", fp, "false negatives:", fn, "all:", err))
		allFp <- allFp + fp 
		allFn <- allFn + fn
	}
	
	print("crossvalidation end")
	
	allFp <- allFp / nrow(data)
	allFn <- allFn / nrow(data)
	acc <- 1 - (allFp+allFn)
	
	return(list("fpr"=allFp, "fnr"=allFn, "acc"=acc))
}

spamfilter.bayes <- function(data, dataIndices, testIndices) {
	model <- naiveBayes(spam ~ ., data, dataIndices)
	results <- predict(model, data[testIndices,-1], threshold=spamfilter.opt.nb.threshold)
	return(results)
}

spamfilter.knn <- function(data, dataIndices, testIndices) {
	
	dist <- matrix()
	
	print("comuputing distance matrix")
	
	if( spamfilter.opt.knn.dist == "cosine") {
		dist <- cosine(t(as.matrix(d[,-1])))
	}else{
		dist <- knn.dist(as.matrix(data[,-1]), dist.meth=spamfilter.opt.knn.dist)
	}
	
	print("predicting")
	results <- knn.predict(dataIndices, testIndices, data[dataIndices,1], dist, k=spamfilter.opt.knn.k)
	return(as.numeric(results))
}

spamfilter.svm <- function(data, dataIndices, testIndices) {
	model <- ksvm(spam ~ ., data[dataIndices,], 
			type=spamfilter.opt.svm.type,
			kernel=spamfilter.opt.svm.kernel,
			C=spamfilter.opt.svm.cost)
	
	results <- predict(model, data[testIndices,-1])
	return(results)
}

spamfilter.test_bayes_threshold<-function(data) {
	old <- spamfilter.opt.nb.threshold
	result <- list()
	
	for(threshold in c(0.5, 0.1, 0.01, 0.001, 0) ) {
		spamfilter.opt.nb.threshold <<- threshold
		error <- spamfilter.crossValidate(data, spamfilter.bayes, spamfilter.opt.folds)
		result[[as.character(threshold)]] <- error
	}
	
	spamfilter.opt.nb.threshold <<- old

	return(result)
}

spamfilter.test_bayes_apriori<-function(data) {
	result <- list()
	
	isSpam <- (data[,1] == 1)
	spam <- data[isSpam,]
	ham <- data[!isSpam,]
	spamCount <- nrow(spam)
	hamCount <- nrow(ham)
	
	if(spamCount > hamCount) {
		print(c("WARNING: this test assumes that there is more spam messages available tham ham!", "spam", spamCount, "ham", hamCount))
		return()
	}
	
	for(ratio in c(5, 3, 2, 1.75, 1.5, 1, 0.5, 0.25)) {
		
		# here we lack more ham data, so we reduce spam data sent
		if( ratio > 2.5) {
			input <- rbind(spam[1:(2*spamCount/ratio), ] , ham[1:2*spamCount,])
		} else {
			input <- rbind(spam, ham[1:spamCount*ratio,])
		}
		
		input <- transform.randomize(input)
		error <- spamfilter.crossValidate(input, spamfilter.bayes, spamfilter.opt.folds)
		result[[as.character(ratio)]] <- error
	}
	
	
	
	return(result)
}

spamfilter.test_knn<-function(data) {
	
	results <- list();
	
	for(dist in c("euclidean", "manhattan", "cosine")) {
		for(k in c(3,7,15)) {
			
			oldk <- spamfilter.opt.knn.k
			oldd <- spamfilter.opt.knn.dist
			
			spamfilter.opt.knn.k <- k
			spamfilter.opt.knn.dist <- dist
			
			error <- spamfilter.crossValidate(data, spamfilter.knn, spamfilter.opt.folds)
			error$k <- k
			error$dist <- dist
			
			results[[paste(dist, k)]] <- error
			
			spamfilter.opt.knn.k <- oldk
			spamfilter.opt.knn.dist <- oldd
		}
	}
	
	return(results)
}

spamfilter.test_svm<-function(data) {
	results <- list();
	
	for(kernel in c("rbfdot", "polydot", "vanilladot")) {
		for(cost in c(0.1, 1, 10)) {
			oldk <- spamfilter.opt.svm.kernel
			oldc <- spamfilter.opt.svm.cost
			
			spamfilter.opt.svm.kernel <- k
			spamfilter.opt.svm.cost <- dist
			
			error <- spamfilter.crossValidate(data, spamfilter.svm, spamfilter.opt.folds)
			error$kernel <- kernel
			error$cost <- cost
			
			results[[paste(kernel, cost)]] <- error
			
			spamfilter.opt.svm.kernel <- oldk
			spamfilter.opt.svm.cost <- oldc
		}
	}
	
	return(results)
}
#problem <- preprocess.do()
#problem$dataset <- transform.toBinary(problem$randomize)
#problem$dataset_binary <- transform.toBinary(problem$dataset)
#problem$dataset_tfidf <- transform.toTfIdf(problem$dataset, problem$dictionary)
#write.table(problem$dataset, "./data.txt")
#write.table(problem$dataset_binary, "./data_bin.txt")
#write.table(problem$dataset_tfidf, "./data_tf.txt")

#d <- read.table("d2.txt")
#d_bin <- transform.toBinary(d)
#d$spam <- factor(d$spam)
#d_bin$spam <- factor(d_bin$spam)

