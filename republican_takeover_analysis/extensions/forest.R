# Honest forests of paired DiD contrasts. Design, not the forest, identifies effects.
.libPaths(c(Sys.getenv('TAKEOVER_R_LIB','/private/tmp/clean-grf-r46'),.libPaths()))
library(grf);library(sandwich)
out <- commandArgs(trailingOnly=TRUE)[1]
dall <- read.csv(file.path(out,'forest_pairs.csv'),stringsAsFactors=FALSE)
save_csv <- function(x,name) write.csv(x,file.path(out,paste0(name,'.csv')),row.names=FALSE)
ci <- function(y,state,high=NULL) {
  f <- if(is.null(high)) lm(y~1) else lm(y~high)
  k <- if(is.null(high)) 1 else 2
  s <- sqrt(vcovCL(f,cluster=state,type='HC1')[k,k]); b <- unname(coef(f)[k]);df <- length(unique(state))-1
  data.frame(estimate=b,se=s,lo=b-qt(.975,df)*s,hi=b+qt(.975,df)*s,p=2*pt(-abs(b/s),df),states=df+1)
}
predictions <- list();validation <- list();support <- list();folds <- list()
for(config in unique(dall$config)) {
  d <- dall[dall$config==config,]; n <- nrow(d)
  historical <- grepl('audit2011',config)
  features <- if(historical) c('log_clean_capacity','log_fossil_capacity','log_core_area','prior_net_mw','prior_net_mw_slope') else
    c('log_clean_capacity','log_fossil_capacity','log_prior_announcements','log_core_area')
  outcomes <- if(historical) 'wind_solar_net_mw' else c('clean_announcements','manufacturing_announcements','wind_solar_net_mw')
  X <- as.matrix(d[features]);stopifnot(all(is.finite(X)))
  set.seed(48123);states <- sample(sort(unique(d$state)));ntest <- max(3,floor(.4*length(states)))
  teststates <- states[seq_len(ntest)];train <- which(!d$state %in% teststates);test <- which(d$state %in% teststates)
  folds[[config]] <- data.frame(config=config,state=states,fold=ifelse(states %in% teststates,'test','train'))
  for(o in outcomes) {
    yR <- d[[paste0(o,'_delta_R')]];yD <- d[[paste0(o,'_delta_D')]]
    fit <- function(idx) causal_forest(X[rep(idx,2),,drop=FALSE],c(yD[idx],yR[idx]),rep(0:1,each=length(idx)),
      Y.hat=rep((yD[idx]+yR[idx])/2,2),W.hat=rep(.5,2*length(idx)),
      clusters=as.integer(factor(rep(d$state[idx],2))),honesty=TRUE,num.trees=4000,min.node.size=4,
      sample.fraction=.5,seed=72311,num.threads=2)
    full <- fit(seq_len(n));raw <- as.numeric(predict(full)$predictions)
    oob <- (raw[seq_len(n)]+raw[n+seq_len(n)])/2;stopifnot(all(is.finite(oob)))
    score <- high <- rep(NA_real_,n)
    status <- 'Insufficient state-disjoint validation support'
    if(n>=25 && length(unique(d$state[train]))>=8 && length(test)>=8) {
      model <- fit(train);score[test] <- predict(model,X[test,,drop=FALSE])$predictions
      cutoff <- median(predict(model)$predictions);high[test] <- as.numeric(score[test]>cutoff)
      ok <- length(unique(high[test]))==2 && min(table(high[test]))>=4 &&
        min(sapply(0:1,function(v)length(unique(d$state[test][high[test]==v]))))>=2
      status <- if(ok) 'Exploratory state-held-out validation' else 'Insufficient high/low subgroup support'
      if(ok) for(v in c('delta','placebo','trend_adjusted')) {
        val <- d[[paste0(o,'_',v)]][test]
        validation[[length(validation)+1]] <- cbind(data.frame(config=config,outcome=o,version=v,
          high_mean=mean(val[high[test]==1]),low_mean=mean(val[high[test]==0]),test_pairs=length(test)),ci(val,d$state[test],high[test]))
      }
    }
    support[[length(support)+1]] <- data.frame(config=config,outcome=o,pairs=n,states=length(states),
      train_pairs=length(train),test_pairs=length(test),train_states=length(unique(d$state[train])),test_states=length(teststates),
      high_pairs=sum(high[test]==1,na.rm=TRUE),low_pairs=sum(high[test]==0,na.rm=TRUE),
      high_states=length(unique(d$state[test][which(high[test]==1)])),low_states=length(unique(d$state[test][which(high[test]==0)])),
      forest_oob_mean=mean(oob),status=status)
    predictions[[length(predictions)+1]] <- data.frame(config=config,outcome=o,pair=d$pair,state=d$state,
      fold=ifelse(seq_len(n) %in% test,'test','train'),forest_oob=oob,heldout_score=score,high=high,
      observed_did=yR-yD,placebo=d[[paste0(o,'_placebo')]])
  }
}
v <- if(length(validation)) do.call(rbind,validation) else data.frame(config=character(),outcome=character(),version=character(),p=numeric())
v$p_holm <- NA_real_
for(c in unique(v$config)) for(ver in unique(v$version)) {
  ix <- v$config==c & v$version==ver;v$p_holm[ix] <- p.adjust(v$p[ix],'holm')
}
save_csv(v,'forest_validation');save_csv(do.call(rbind,predictions),'forest_predictions')
save_csv(do.call(rbind,support),'forest_support');save_csv(do.call(rbind,folds),'forest_folds')
writeLines(capture.output(sessionInfo()),file.path(out,'R_session.txt'))
print(do.call(rbind,support));print(v)
