library('RPostgreSQL')
library(ggplot2)
library(pROC)
library(caret)

args = commandArgs(trailingOnly=TRUE)

output_folder = args[1]

#create connection object
con <- dbConnect(drv =PostgreSQL(), 
                 user=args[4], 
                 password=args[5],
                 host=args[6], 
                 port=5432, 
                 dbname=args[7])

#dbListTables(con)   #list all the tables 

#query the database and store the data in datafame
sql = gsub(".?table_name", paste("\"",args[2],"\"",sep=""), args[3])
#sql = gsub("??", paste("\"","","",sep=""), sql)
#str_replace(sql, "??", "\"")
# sql = gsub("??", "\"", sql)
print('THE SQL statement:')
print(sql)
df <- dbGetQuery(con, sql)
dbDisconnect(con)

size_data = nrow(df)
df$score  <- "Value"
rv <- vector()

for(i in 1:size_data){
  prob = as.double(substr(sub(".*;1:", "", df$confidences[i]), 1, 6))
  rv <- c(rv, prob)
  }
  # first_results$confidences_temp[i] = prob

df$score <- rv

# ROC Curve
pdf(file=file.path(output_folder, 'roc.pdf'))

my_roc = roc(df$labels_transformed, df$score, smoothed = TRUE,
   # arguments for plot
   plot=TRUE,
   print.auc=TRUE, show.thres=TRUE)
my_roc
youden_index = coords(my_roc, "best")
dev.off() 

categories = c('LCA', 'RCA')
d1 <- data.frame(Col1=c(0, 1), Col2=categories, stringsAsFactors=FALSE)
df$labels_transformed = d1$Col2[match(df$labels_transformed, d1$Col1)]

pdf(file=file.path(output_folder, 'histogram.pdf'))

ggplot(df, aes(x=score, color=labels_transformed)) +
  geom_histogram(fill="white")
dev.off() 

# confusion matrix


#create confusion matrix
confmat <- confusionMatrix(data=factor(d1$Col2[match(df$predictions, d1$Col1)]), reference = factor(df$labels_transformed))
draw_confusion_matrix <- function(cm, cats) {
  
  layout(matrix(c(1,1,2)))
  par(mar=c(2,2,2,2))
  plot(c(100, 345), c(300, 450), type = "n", xlab="", ylab="", xaxt='n', yaxt='n')
  title('CONFUSION MATRIX', cex.main=2)
  
  # create the matrix 
  rect(150, 430, 240, 370, col='#3F97D0')
  text(195, 435, cats[1], cex=1.2)
  rect(250, 430, 340, 370, col='#F7AD50')
  text(295, 435, cats[2], cex=1.2)
  text(125, 370, 'Predicted', cex=1.3, srt=90, font=2)
  text(245, 450, 'Actual', cex=1.3, font=2)
  rect(150, 305, 240, 365, col='#F7AD50')
  rect(250, 305, 340, 365, col='#3F97D0')
  text(140, 400, cats[1], cex=1.2, srt=90)
  text(140, 335, cats[2], cex=1.2, srt=90)
  
  # add in the cm results 
  res <- as.numeric(cm$table)
  text(195, 400, res[1], cex=1.6, font=2, col='white')
  text(195, 335, res[2], cex=1.6, font=2, col='white')
  text(295, 400, res[3], cex=1.6, font=2, col='white')
  text(295, 335, res[4], cex=1.6, font=2, col='white')
  
  # add in the specifics 
  plot(c(100, 0), c(100, 0), type = "n", xlab="", ylab="", main = "DETAILS", xaxt='n', yaxt='n')
  text(10, 85, names(cm$byClass[1]), cex=1.2, font=2)
  text(10, 70, round(as.numeric(cm$byClass[1]), 3), cex=1.2)
  text(30, 85, names(cm$byClass[2]), cex=1.2, font=2)
  text(30, 70, round(as.numeric(cm$byClass[2]), 3), cex=1.2)
  text(50, 85, names(cm$byClass[5]), cex=1.2, font=2)
  text(50, 70, round(as.numeric(cm$byClass[5]), 3), cex=1.2)
  text(70, 85, names(cm$byClass[6]), cex=1.2, font=2)
  text(70, 70, round(as.numeric(cm$byClass[6]), 3), cex=1.2)
  text(90, 85, names(cm$byClass[7]), cex=1.2, font=2)
  text(90, 70, round(as.numeric(cm$byClass[7]), 3), cex=1.2)
  
  # add in the accuracy information 
  text(30, 35, names(cm$overall[1]), cex=1.5, font=2)
  text(30, 20, round(as.numeric(cm$overall[1]), 3), cex=1.4)
  text(70, 35, names(cm$overall[2]), cex=1.5, font=2)
  text(70, 20, round(as.numeric(cm$overall[2]), 3), cex=1.4)
}

draw_confusion_matrix_simple <- function(cm, cats) {
  
  layout(matrix(c(1,1,2)))
  par(mar=c(2,2,2,2))
  plot(c(100, 345), c(300, 450), type = "n", xlab="", ylab="", xaxt='n', yaxt='n')
  title('CONFUSION MATRIX', cex.main=2)
  
  # create the matrix 
  rect(150, 430, 240, 370, col='#3F97D0')
  text(195, 435, cats[1], cex=1.2)
  rect(250, 430, 340, 370, col='#F7AD50')
  text(295, 435, cats[2], cex=1.2)
  text(125, 370, 'Predicted', cex=1.3, srt=90, font=2)
  text(245, 450, 'Actual', cex=1.3, font=2)
  rect(150, 305, 240, 365, col='#F7AD50')
  rect(250, 305, 340, 365, col='#3F97D0')
  text(140, 400, cats[1], cex=1.2, srt=90)
  text(140, 335, cats[2], cex=1.2, srt=90)
  
  # add in the cm results 
  res <- as.numeric(cm$table)
  text(195, 400, res[1], cex=1.6, font=2, col='white')
  text(195, 335, res[2], cex=1.6, font=2, col='white')
  text(295, 400, res[3], cex=1.6, font=2, col='white')
  text(295, 335, res[4], cex=1.6, font=2, col='white')
}

pdf(file=file.path(output_folder, 'conf_matrix_ext.pdf'))
draw_confusion_matrix(confmat, categories)
dev.off() 

pdf(file=file.path(output_folder, 'conf_matrix.pdf'))
draw_confusion_matrix_simple(confmat, categories)
dev.off()
