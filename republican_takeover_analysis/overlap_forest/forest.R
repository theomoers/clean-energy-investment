.libPaths(c(Sys.getenv("TAKEOVER_R_LIB","/private/tmp/clean-grf-r46"),.libPaths()))
library(grf)
out <- commandArgs(trailingOnly=TRUE)[1]
d <- read.csv(file.path(out,"areas.csv"),stringsAsFactors=FALSE)
modifiers <- c("log_clean_capacity","log_fossil_capacity","log_prior_announcements","log_core_area")
features <- c(modifiers,"prior_announcement_slope","log_prior_manufacturing",
              "prior_manufacturing_slope","prior_net_mw","prior_net_mw_slope")
outcomes <- c("clean_announcements","manufacturing_announcements","wind_solar_net_mw")
X <- as.matrix(d[modifiers])
train <- which(d$fold=="train");test <- which(d$fold=="test")
stopifnot(length(intersect(unique(d$state[train]),unique(d$state[test])))==0)
weighted_median <- function(x,w) {i<-order(x);x[i][which(cumsum(w[i])/sum(w)>=.5)[1]]}
fit <- function(idx,o,prop) {
  y <- d[[paste0(o,"_delta")]][idx]
  # Outcome model includes all baseline controls and state indicators.
  Z <- model.matrix(~.-1,data=d[idx,c(features,"state")])
  cl <- as.integer(factor(d$state[idx]))
  nuisance <- regression_forest(Z,y,clusters=cl,num.trees=4000,
    min.node.size=5,honesty=TRUE,sample.fraction=.5,seed=91311,num.threads=2)
  m <- as.numeric(predict(nuisance)$predictions)
  stopifnot(all(is.finite(m)))
  forest <- causal_forest(X[idx,,drop=FALSE],y,d$gainer[idx],
    Y.hat=m,W.hat=prop,clusters=cl,num.trees=4000,min.node.size=5,
    honesty=TRUE,sample.fraction=.5,seed=72311,num.threads=2)
  list(forest=forest,m=m)
}
predictions <- list();averages <- list();nuisances <- list();details <- list()
for(o in outcomes) {
  full <- fit(seq_len(nrow(d)),o,d$propensity)
  fullpred <- predict(full$forest)$predictions
  native <- average_treatment_effect(full$forest,target.sample="overlap")
  G <- length(unique(d$state));crit<-qt(.975,G-1)
  averages[[o]] <- data.frame(outcome=o,estimate=native[1],se=native[2],
    lo=native[1]-crit*native[2],hi=native[1]+crit*native[2],
    p=2*pt(-abs(native[1]/native[2]),G-1),states=G,
    interpretation="Model-assisted overlap score; SE conditional on nuisance fits")
  training <- fit(train,o,d$training_propensity[train])
  oob <- as.numeric(predict(training$forest)$predictions)
  p <- d$training_propensity[train]
  cutoff <- weighted_median(oob,p*(1-p))
  scores <- rep(NA_real_,nrow(d));high <- scores
  scores[test] <- predict(training$forest,X[test,,drop=FALSE])$predictions
  high[test] <- as.numeric(scores[test]>cutoff)
  stopifnot(all(is.finite(fullpred)),all(is.finite(scores[test])))
  predictions[[o]] <- data.frame(unit=d$unit,state=d$state,fold=d$fold,outcome=o,
    full_oob_prediction=fullpred,heldout_score=scores,high=high,training_cutoff=cutoff)
  nuisances[[o]] <- data.frame(unit=d$unit,outcome=o,outcome_oob=full$m,
    propensity=d$propensity,residual_y=d[[paste0(o,"_delta")]]-full$m,
    residual_w=d$gainer-d$propensity)
  details[[o]] <- data.frame(outcome=o,train_areas=length(train),test_areas=length(test),
    train_states=length(unique(d$state[train])),test_states=length(unique(d$state[test])),
    cutoff=cutoff,high=sum(high[test]==1),low=sum(high[test]==0),
    training_score_sd=sd(oob),heldout_score_sd=sd(scores[test]))
  print(details[[o]])
}
write.csv(do.call(rbind,predictions),file.path(out,"forest_predictions.csv"),row.names=FALSE)
write.csv(do.call(rbind,averages),file.path(out,"native_overlap_score.csv"),row.names=FALSE)
write.csv(do.call(rbind,nuisances),file.path(out,"nuisance_predictions.csv"),row.names=FALSE)
write.csv(do.call(rbind,details),file.path(out,"forest_support.csv"),row.names=FALSE)
writeLines(capture.output(sessionInfo()),file.path(out,"R_session.txt"))
