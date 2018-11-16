ApplyTransformations <- function (data) {
  data$loan_amnt=log(data$loan_amnt)
  data$annual_inc=ifelse(data$annual_inc==0,0.00001,data$annual_inc)
  data$revol_bal=ifelse(data$revol_bal==0,0.00001,data$revol_bal)
  data$annual_inc=log(data$annual_inc)
  data$term=as.numeric(data$term)
  data$earliest_cr_line=substr(data$earliest_cr_line,5,8)
  data$fico=0.5*data$fico_range_low + 0.5*data$fico_range_high
  data$revol_util[is.na( data$revol_util)] = 0
  # data$home_ownership=ifelse(data$home_ownership=='ANY')
  #ifelse(data.train$loan_status=='Fully Paid', 1, 0)
  data$revol_bal=log(data$revol_bal)
  data$mort_acc[is.na( data$mort_acc)] = 0
  data$pub_rec_bankruptcies[is.na( data$pub_rec_bankruptcies)] = 0
  data$dti[is.na(data$dti)] = 0
  data$fico=(data$fico)
  data$mort_acc=(data$mort_acc)
  data=subset(data,
              select = -c(title,emp_length,emp_title,zip_code,fico_range_low,fico_range_high,
                          grade,application_type,initial_list_status,addr_state) )
  
  
  return(data)
}

convertToNumericMatrix  = function (df) {
  return_matrix = matrix(0, nrow = nrow(df), ncol = ncol(df))
  
  for (i in 1:ncol(df)) {
    return_matrix[,i] = as.numeric(df[,i])
  }
  
  return(return_matrix)
}


for (i in 1:1){
  data.train = read.csv(paste0("train-",i,".csv"))
  data.train$loan_status=ifelse(data.train$loan_status=='Fully Paid', 1, 0)
  data.train=ApplyTransformations(data.train)
  data.train.X = convertToNumericMatrix(subset(data.train,select=-c(loan_status,id)))
  label.x=data.train$loan_status
  cv.out=cv.glmnet(data.train.X,label.x,family="binomial")
  
  data.test = read.csv(paste0("test-",i,".csv"))
  data.test=ApplyTransformations(data.test)
  data.test.X = convertToNumericMatrix(subset(data.test,select=-c(loan_status,id)))
  y_pre_glm =predict(cv.out,data.test.X,lambda=cv.out$lambda.1se,type="response")
  mysubmission=as.data.frame(cbind(data.test$id, y_pre_glm ))
  colnames(mysubmission) = c("id", "prob")
  write.csv(mysubmission, file = paste0("mysubmission-",i,".txt"), row.names = FALSE, quote = FALSE)
  
  ##
  ev=as.data.frame(cbind(y_pre_glm,ifelse(data.test$loan_status=='Fully Paid', 1, 0)))
  colnames(ev)=c('prob','y')
  print(with(ev, logLoss(y, prob)))
  
}



