
#----------------------------------------------------------------------------------------
# Libs
#----------------------------------------------------------------------------------------
st.script = Sys.time() #track script runtime
library(tidyverse)
library(magrittr)
library(stringr)
library(Boruta)
library(caret)
library(xgboost)
library(glmnet)
library(e1071)
library(doParallel)
library(foreach)

#----------------------------------------------------------------------------------------
# Read in/clean data
#----------------------------------------------------------------------------------------
#read all saved BigQuery data from file
#*********************
charges_2012 = read.csv("medicare_data/cms_medicare_inpatient_charges_2012.csv") %>% mutate(charge_year=2012)
charges_2013 = read.csv("medicare_data/cms_medicare_inpatient_charges_2013.csv") %>% mutate(charge_year=2013)
charges_2014 = read.csv("medicare_data/cms_medicare_inpatient_charges_2014.csv") %>% mutate(charge_year=2014)
charges_2015 = read.csv("medicare_data/cms_medicare_inpatient_charges_2015.csv") %>% mutate(charge_year=2015)
charges = rbind(charges_2012, charges_2013, charges_2014, charges_2015) %>% mutate(charge_year = as.factor(charge_year))

hospitals = read.csv("medicare_data/cms_medicare_hospital_general_info.csv")

race = read.csv("medicare_data/race.csv") %>% select(state, state_prct_minority)

#*********************
# Clean charges
#*********************
#drop unused columns
charges %<>% mutate(provider_name=NULL, 
                    provider_street_address=NULL,
                    provider_city=NULL,
                    provider_zipcode=NULL,
                    provider_name=NULL,
                    hospital_referral_region_description = NULL, #powerful predictor, but removed due to compute cost due to high #levels
                    average_covered_charges=NULL,
                    average_medicare_payments=NULL)

#create col for ms-drg numeric id only
charges %<>% mutate(drg_num=str_sub(str_trim(drg_definition), 1, 3),
                    drg_num = as.numeric(drg_num))

#put cols in sensible order
charges %<>% select(provider_id, drg_num, drg_definition, average_total_payments, everything())


#*********************
# Clean hospitals
#*********************
#drop footnote columns
hospitals %<>% select(-ends_with("footnote"))

#drop unused columns
#NOTE: we drop state here because it already exists in 'charges'
hospitals %<>% mutate(hospital_name=NULL, 
                      address=NULL,
                      city=NULL,
                      state=NULL,
                      zip_code=NULL,
                      county_name = NULL,
                      phone_number=NULL,
                      location=NULL)

#column below had several NA values, all others were TRUE, so set these to FALSE
hospitals %<>% mutate(meets_criteria_for_meaningful_use_of_ehrs = 
                        ifelse(is.na(meets_criteria_for_meaningful_use_of_ehrs), 
                               FALSE, 
                               meets_criteria_for_meaningful_use_of_ehrs))

#convert these 2 logical vars to factors
hospitals %<>% mutate(emergency_services = as.factor(emergency_services),
                      meets_criteria_for_meaningful_use_of_ehrs = as.factor(meets_criteria_for_meaningful_use_of_ehrs))


#*********************
# Combine charges, hospitals, census data
#*********************
combined = inner_join(charges, hospitals, by="provider_id") %>% 
  inner_join(race, by=c("provider_state"="state")) %>%
  mutate(provider_state=as.factor(provider_state))
#check for missing data
#colnames(combined)[apply(combined, 2, function(x) any(is.na(x) | is.infinite(x)))]


#----------------------------------------------------------------------------------------
# Functions to build models and calculate model metrics
#----------------------------------------------------------------------------------------
# Generates all model metrics
# Params:
#     p.results     => df containing model perf on test data with columns: actual/predicted/residuals
#     train         => train data
#     test          => test data
#     num_features  => total features used in model (may differ from nrow(train) - for example lasso will 0 out some coef)
#     rmse_guess    => rmse when using mean of response variable from the train data for all predictions on test data
#     st            => start time of model execution (used to calculate total runtime)
#     msdrg         => the specific drg being processed
# Returns: dataframe with various model metrics (e.g. R2, R2_Adjusted, RMSE)
calc_model_metrics = function(p.results, train, test, num_features, rmse_guess, st, msdrg)
{
  #calc results
  p.perf = postResample(pred = p.results$pred, obs = p.results$actual)
  rmse = p.perf[1]
  rmse_prct_chg_from_guess_to_model = round((rmse-rmse_guess)/(rmse_guess)*100,0) #negative number means improvement
  r2 = p.perf[2]
  num_train_obs = nrow(train)
  num_test_obs = nrow(test)
  #https://www.tutorialspoint.com/statistics/adjusted_r_squared.htm
  r2_adj = 1-(((1-r2)*(num_test_obs-1))/(num_test_obs-num_features-1))
  runtime_min = round(difftime(Sys.time(), st, units="min"),2)
  
  #store in df to return
  df = data.frame(drg_num=msdrg$drg_num[1],
                  drg_definition=as.character(msdrg$drg_definition[1]),
                  r2,
                  r2_adj,
                  rmse_guess,
                  rmse,
                  rmse_prct_chg_from_guess_to_model,
                  num_features,
                  num_train_obs,
                  num_test_obs,
                  runtime_min,
                  stringsAsFactors = FALSE,
                  row.names = NULL)
  return(df)
}

# Build models and perform feature analysis using xgboost, glmnet+lasso, and svm
# Params:
#     msdrg         => the specific drg to be modeled
# Returns: list of dataframes containing results of each model or feature importance analysis
run_models_and_feature_analysis = function(msdrg)
{
  #identify 1 level factors in both train/test and combine, we remove these prior to running
  # model.matrix or it throws an error
  #NOTE: these are constants so no good to the model anyway, no variation
  one_level_factors = unique(c(msdrg %>% select_if(~nlevels(.)==1) %>% colnames()))
  
  #tt = abbr for train/test
  tt = msdrg %>% select(-c(1:3), -one_of(one_level_factors))
  
  #split tt into 70% train and 30% test
  set.seed(7)
  train_ids = sample(seq(1:nrow(tt)), round(nrow(tt)*.70,0))
  
  #svm format
  train = tt[train_ids,]
  test = tt[-train_ids,]
  
  #xgboost and lasso format (one hot encoded)
  tt.one_hot = model.matrix(~.-1, data = tt)
  train.x = tt.one_hot[train_ids,-1]
  train.y = tt.one_hot[train_ids,1]
  test.x = tt.one_hot[-train_ids,-1]
  test.y = tt.one_hot[-train_ids,1]
  
  
  #*********************
  #feature analysis with Boruta
  #*********************
  set.seed(7)
  st.boruta = Sys.time()
  boruta.train = Boruta(average_total_payments~., data=train, doTrace=0)
  boruta.train = TentativeRoughFix(boruta.train)
  rt.boruta = difftime(Sys.time(), st.boruta, units="min")
  
  rs.boruta.feature_importance = attStats(boruta.train) %>%
    mutate(Feature = row.names(.),
           drg_num = msdrg$drg_num[1],
           drg_definition = msdrg$drg_definition[1],
           runtime_min = round(rt.boruta,2)) %>%
    select(drg_num, drg_definition, everything()) %>%
    arrange(desc(meanImp))
  
  
  #*********************
  #Benchmark by calculating RMSE using 'best guess' (without having a model).
  #In this case 'best guess' would be using mean of train data to predict charges found in test data.
  #We can use this as a general idea of how well our model predicts compared with 'no model' at all.
  #*********************
  guess = mean(train.y)
  guess.results = cbind(pred=guess, actual=test.y) %>% data.frame() %>% mutate(residual=actual-pred)
  rmse_guess = postResample(guess.results$pred, guess.results$actual)[1]
  
  
  #*********************
  # xgboost
  #*********************
  set.seed(7)
  st = Sys.time()
  cv = xgb.cv(data = train.x,
              label = train.y,
              nthread = detectCores(),
              nrounds = 5000,
              early_stopping_rounds = 100,
              nfold = 10,
              objective = "reg:linear",
              eval_metric = "rmse",
              verbose = FALSE)
  
  m = xgboost(data = train.x,
              label = train.y,
              nthread = detectCores(),
              nrounds = cv$best_iteration,
              objective = "reg:linear",
              verbose = FALSE)
  
  #predict
  p = predict(m, test.x)
  p.results = cbind(pred=p, actual=test.y) %>% data.frame() %>% mutate(residual=actual-pred)
  
  #calc results and store in df
  #NOTE: ran into this issue for num_features https://www.kaggle.com/c/homesite-quote-conversion/discussion/18669
  #      to handle for, we use xgb.importance vs m$feature_names to exclude any features not used
  #num_features = m$nfeatures #NOT USING
  num_features = nrow(xgb.importance(m$feature_names, m)) #still debating this.. should go back to use ALL features?
  rs.xgb = calc_model_metrics(p.results, train, test, num_features, rmse_guess, st, msdrg)
  
  #calc_model_metrics does not account for xgboost num_trees, so added at end here
  rs.xgb = cbind(rs.xgb, num_trees = m$niter)
  
  #xgboost feature importance
  #REF: https://towardsdatascience.com/be-careful-when-interpreting-your-features-importance-in-xgboost-6e16132588e7
  #above link refers to classification, but assume metric desc would hold true for regression as well, so considering
  # gain as the most relevant metric here. Refer to above link for insight on that metric.
  rs.xgb.feature_importance = xgb.importance(m$feature_names, m) %>% 
    data.frame() %>%
    mutate(drg_num=msdrg$drg_num[1],
           drg_definition=as.character(msdrg$drg_definition[1])) %>%
    select(drg_num, drg_definition, everything()) %>%
    arrange(desc(Gain))
  
  
  #*********************
  # glmnet+lasso
  #*********************
  rm(m, cv, p, p.results)
  set.seed(7)
  st = Sys.time()
  cv = cv.glmnet(train.x, train.y, alpha = 1)
  m = glmnet(train.x, train.y, alpha = 1, lambda = cv$lambda.min)
  
  #predict
  p = predict(m, test.x)
  p.results = cbind(pred=p[,], actual=test.y) %>% data.frame() %>% mutate(residual=actual-pred)
  
  #get non 0 coef (these are the features 'kept' by lasso), which will use to determine num_features
  #https://gist.github.com/ydavidchen/a166bc364cfadd53921bb9a6f07100bb
  myCoefs <- coef(m)
  non_zero_coef <- data.frame(
    features = myCoefs@Dimnames[[1]][ which(myCoefs != 0 ) ], #intercept included
    coefs    = myCoefs              [ which(myCoefs != 0 ) ]  #intercept included
  )[-1,] #-1 here drops intercept which we dont want to count as a feature
  
  #calc results and store in df
  num_features = nrow(non_zero_coef)
  rs.lasso = calc_model_metrics(p.results, train, test, num_features, rmse_guess, st, msdrg)
  
  #lasso feature importance (coef magnitude used to identify this.. bigger coef = bigger impact on response var)
  rs.lasso.feature_importance = non_zero_coef %>% 
    mutate(coefs = abs(coefs),
           drg_num=msdrg$drg_num[1],
           drg_definition=as.character(msdrg$drg_definition[1])) %>%
    select(drg_num, drg_definition, everything()) %>%
    rename(Feature = features,
           coef_abs_val = coefs) %>%
    arrange(desc(coef_abs_val))
  
  
  #*********************
  #svm
  #*********************
  rm(m, cv, p, p.results)
  set.seed(7)
  st = Sys.time()
  
  #This tuning step runs 25 models to find best gamma/cost params - this step has signficant impact on model accuracy
  #NOTE: lamens detail on gamma/cost params: https://www.datasciencecentral.com/profiles/blogs/svm-in-practice
  #May need to consider running tuning in parallel with dopar (could also increase tune range going that route)
  #tune = tune.svm(average_total_payments~., data=train, gamma = 2^(-2:2), cost = 2^(1:5))
  #m = svm(average_total_payments~., train, gamma=tune$best.parameters[1], cost=tune$best.parameters[2])
  
  #*************** MOD to run SVM in parallel **************
  #build grid of gamma/cost to find best values
  #gamma = 2^(-2:2)
  gamma = 2^(-4:0)
  cost = 2^(1:5)
  params = expand.grid(cost = cost, gamma = gamma)
  
  #set-up parallel processing
  num_cores = detectCores()    #detect cores on this machine, some set this to -1 to leave one unused
  cl = makeCluster(num_cores)  #required for Windows
  registerDoParallel(cl)       #finalize set-up
  
  #function to calculate error for each value in params grid (in this case params are gamma and cost)
  #each iteration is done in parallel on a different core - this is why we have to load e1071 each time
  get_results = function(params, train)
  {
    library(e1071)
    tune = tune.svm(average_total_payments~., data=train, gamma = params$gamma, cost = params$cost)
    return(tune$performances)
  }
  
  #process results of each set of params in grid, this is done in parallel via %dopar%
  results = foreach(i=1:nrow(params)) %dopar% get_results(params[i,], train)
  
  #stop the cluster
  stopCluster(cl)
  
  #convert results into a dataframe and arrange with min error (MSE used by default) at top for best gamma/cost values
  results.best = bind_rows(results) %>% filter(error==min(error))
  gamma.best = results.best$gamma
  cost.best = results.best$cost
  
  m = svm(average_total_payments~., train, gamma=gamma.best, cost=cost.best)
  #***********************************************
  
  #predict
  p = predict(m, test)
  p.results = cbind(pred=p, actual=test[,1]) %>% data.frame() %>% mutate(residual=actual-pred)
  
  #calc results and store in df
  num_features = ncol(train)
  rs.svm = calc_model_metrics(p.results, train, test, num_features, rmse_guess, st, msdrg)
  
  #calc_model_metrics does not account for svm's gamma/cost, so add here
  rs.svm = cbind(rs.svm, gamma=gamma.best)
  rs.svm = cbind(rs.svm, cost=cost.best)
  
  
  #*********************
  # Return key dataframes
  #*********************
  return(list(boruta_features=rs.boruta.feature_importance, 
              xgboost_results=rs.xgb,
              xgboost_features=rs.xgb.feature_importance,
              lasso_results=rs.lasso,
              lasso_features=rs.lasso.feature_importance,
              svm_results=rs.svm))

}


#----------------------------------------------------------------------------------------
# Iterate over drgs and build various models/feature analysis for each
# Writes final outputs to file
#----------------------------------------------------------------------------------------
#define dfs to store accumulated model results for all drgs
rs.boruta_features = data.frame()
rs.xgboost_results = data.frame()
rs.xgboost_features = data.frame()
rs.lasso_results = data.frame()
rs.lasso_features = data.frame()
rs.svm_results = data.frame()

#A key concept we use below is to isolate drgs within a specific Major Diagnostic Category (MDC)
#MDC REF: https://en.wikipedia.org/wiki/Major_Diagnostic_Category
#lets set the mdc to: Diseases and Disorders of the circulatory System 215 - 316
mdc = combined %>% filter(drg_num >= 215 & drg_num <= 316)

#summarize drg counts within this mdc
mdc_drgs = mdc %>%
  group_by(drg_num, drg_definition) %>% 
  count() %>% 
  rename(obs=n) %>%
  arrange(desc(obs)) %>%
  ungroup()

#set a threshold for min observations accepted to attempt a model
min_obs_threshold = 100

#log the drgs we are skipping due to not having enough observations
drgs_skipped_limited_observations = mdc_drgs %>% filter(obs < min_obs_threshold)

#subset to only those drgs in scope for modeling
mdc_drgs %<>% filter(obs >= min_obs_threshold)
#TESTING ---------------
#mdc_drgs %<>% filter(drg_num %in% c(220, 219)) #run on 2 low obs drgs - good for ensuring basic functionality working
#mdc_drgs %<>% filter(drg_num %in% c(292))      #largest single drg
#mdc_drgs %<>% filter(obs < 5500 & obs > 100)   #run on lower half of all drgs within MDC 
#END TESTING -----------

#iterate through each drg and build models
for(i in 1:nrow(mdc_drgs))
{
  #build models, metrics, and feature importance
  msdrg = mdc %>% filter(drg_num==mdc_drgs$drg_num[i]) %>% droplevels() 
  drg_results = run_models_and_feature_analysis(msdrg)
  
  #store results
  rs.boruta_features = rbind(rs.boruta_features, drg_results$boruta_features)
  rs.xgboost_results = rbind(rs.xgboost_results, drg_results$xgboost_results)
  rs.xgboost_features = rbind(rs.xgboost_features, drg_results$xgboost_features)
  rs.lasso_results = rbind(rs.lasso_results, drg_results$lasso_results)
  rs.lasso_features = rbind(rs.lasso_features, drg_results$lasso_features)
  rs.svm_results = rbind(rs.svm_results, drg_results$svm_results)
  
  #track real-time script progess, writes empty log file with drg name to directory that can be monitored
  write.csv("", paste0("runtime_tracking/", mdc_drgs$drg_num[i], ".csv"))
}

#write final output to file
time_stamp = str_replace_all(str_replace_all(Sys.time(), ":", "-"), " ", "_")
write.csv(rs.boruta_features, paste0("output/boruta_features_", time_stamp, ".csv"), row.names = FALSE)
write.csv(rs.xgboost_results, paste0("output/xgboost_results_", time_stamp, ".csv"), row.names = FALSE)
write.csv(rs.xgboost_features, paste0("output/xgboost_features_", time_stamp, ".csv"), row.names = FALSE)
write.csv(rs.lasso_results, paste0("output/lasso_results_", time_stamp, ".csv"), row.names = FALSE)
write.csv(rs.lasso_features, paste0("output/lasso_features_", time_stamp, ".csv"), row.names = FALSE)
write.csv(rs.svm_results, paste0("output/svm_results_", time_stamp, ".csv"), row.names = FALSE)
write.csv(drgs_skipped_limited_observations, paste0("output/drgs_skipped_limited_observations_", time_stamp, ".csv"), row.names = FALSE)
write.csv(difftime(Sys.time(), st.script, units="min"), paste0("output/script_runtime_minutes_", time_stamp, ".csv"), row.names = FALSE)

#print script runtime
difftime(Sys.time(), st.script, units="min")


