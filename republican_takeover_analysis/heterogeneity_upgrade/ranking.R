.libPaths(c(Sys.getenv("TAKEOVER_R_LIB","/private/tmp/clean-grf-r46"),.libPaths()))
library(grf)
out <- commandArgs(trailingOnly=TRUE)[1]
d <- read.csv(file.path(out,"areas.csv"),stringsAsFactors=FALSE)
mods <- c("log_clean_capacity","log_fossil_capacity","log_prior_announcements","log_core_area")
features <- c(mods,"prior_announcement_slope","log_prior_manufacturing",
              "prior_manufacturing_slope","prior_net_mw","prior_net_mw_slope")
train <- which(d$fold=="train");test <- which(d$fold=="test")
X <- as.matrix(d[mods]);Z<-model.matrix(~.-1,data=d[train,c(features,"state")])
cl<-as.integer(factor(d$state[train]));rows<-list()
for(o in c("clean_announcements","manufacturing_announcements","wind_solar_net_mw")) {
  y<-d[[paste0(o,"_delta")]][train]
  nm<-regression_forest(Z,y,clusters=cl,num.trees=4000,min.node.size=5,
    honesty=TRUE,sample.fraction=.5,seed=91311,num.threads=2)
  cf<-causal_forest(X[train,,drop=FALSE],y,d$gainer[train],
    Y.hat=predict(nm)$predictions,W.hat=d$training_propensity[train],
    clusters=cl,num.trees=4000,min.node.size=5,honesty=TRUE,sample.fraction=.5,
    seed=72311,num.threads=2)
  f<-rep(NA_real_,nrow(d));f[train]<-predict(cf)$predictions
  f[test]<-predict(cf,X[test,,drop=FALSE])$predictions
  rows[[o]]<-data.frame(unit=d$unit,outcome=o,priority=f,fold=d$fold)
}
write.csv(do.call(rbind,rows),file.path(out,"priorities.csv"),row.names=FALSE)
writeLines(capture.output(sessionInfo()),file.path(out,"R_session.txt"))
