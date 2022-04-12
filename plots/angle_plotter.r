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

df <- dbGetQuery(con, sql)
dbDisconnect(con)

size_data = nrow(df)


ggplot(df, aes(x=wt, y=mpg, shape=cyl, color=cyl)) +
  geom_point()
