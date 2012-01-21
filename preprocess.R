# setup console
options(width=1000)

#libraries
library("Snowball") # stemmer
library("kernlab")	# svm

# preprocessing options
preprocess.options.stemming <- TRUE
preprocess.options.minimumTermLength <- 3
preprocess.options.minimumTermDocuments <- 10
preprocess.options.stopwords <- c("i", "a", "about", "an", "are", "as", "at", "be", "by", "com", "for", 
		"from", "how", "in", "is", "it", "of", "on", "or", "tha", "the ", "thi", 
		"to", "was", "what", "when", "where", "who", "will", "with", "the", "www")

preprocess.readDirectories<-function(dir) {
	
	result<-c()
	for(filename in list.files(dir)){
		path<-paste(dir, filename, sep="")
		fileHandle<-file(path, "rt")
		conent<-readChar(fileHandle, 1000000)
		result<-append(result, conent)
		close(fileHandle)
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
		}
	}
	
	return(result)
}

preprocess.buildDictionary<-function(documents) {
	
	dictionary <- list()

	for(document in documents) {
		
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
	
	# filter all stopwords
	for(word in preprocess.options.stopwords) {
		result[[word]] <- NULL
	}
	
	# filter all words of too small length or frequency
	for(word in names(dictionary)) {
		if(nchar(word) < preprocess.options.minimumTermLength || dictionary[[word]] < preprocess.options.minimumTermDocuments) {
			result[[word]] <- NULL
		}
	}
	
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
	colnames(result) <- c(".isSpam", names(dictionary))  #c("isSpam", paste("word.", names(dictionary), sep=""))
	result <- as.data.frame(result)
	
	# fill rows
	for(di in 1:length(documents)) {
		document <- documents[[di]]
		result[[1]][di] <- classes[di]
		
		for(term in document) {
			if(!is.null(result[[term]])) {
				result[[term]][di] <- result[[term]][di] + 1
			}
		}
	}
	
	emptyRows <- c()
	if(columns > 1) {
		for(ri in 1:length(documents)) {
			record <- result[ri,-1]
			if(all(record == 0)) {
				emptyRows <- c(emptyRows, ri)
			}
		}
	}

	if(!is.null(emptyRows)) {
		print(c("WARNING: REMOVING EMPTY DOCUMENTS FROM SET", emptyRows))
		result = emptyRows[-emptyRows,]
	}
	
	return(result)
}

preprocess.do <- function() {
	
	problem = list()
	
	print("[preprocess] [1/6] reading directories")
	problem$rawDocuments <- preprocess.readDirectories("./data/test/")
	problem$rawClasses <- rep.int(1, length(problem$rawDocuments))
	
	print("[preprocess] [2/6] extracting terms")
	problem$documents <- preprocess.extractTerms(problem$rawDocuments)
	
	print("[preprocess] [3/6] building dictionary")
	problem$dictionary <- preprocess.buildDictionary(problem$documents)
	
	print("[preprocess] [4/6] filtering dictionary")
	problem$dictionary <- preprocess.filterDictionary(problem$dictionary)
	
	print("[preprocess] [5/6] building dataset")
	problem$dataset <- preprocess.buildData(problem$documents, problem$rawClasses, problem$dictionary);
	
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

problem <- preprocess.do()
problem$dataset_binary <- transform.toBinary(problem$dataset)
problem$dataset_tfidf <- transform.toTfIdf(problem$dataset, problem$dictionary)


