# Honest state-clustered forests of within-state paired changes.
# The midpoint transformation makes the target the paired DiD exactly.
# W.hat=.5 is a within-pair weighting identity, not random party assignment.
.libPaths(c(Sys.getenv("TAKEOVER_R_LIB", "/private/tmp/clean-grf-r46"),.libPaths()))
library(grf)
library(sandwich)
args <- commandArgs(trailingOnly=TRUE)
out <- args[1]
dat <- read.csv(file.path(out,"pairs.csv"),stringsAsFactors=FALSE)
features <- c("log_clean_capacity","log_fossil_capacity","log_prior_announcements","log_core_area")
outcomes <- c("clean_announcements","manufacturing_announcements","wind_solar_net_mw")
save_csv <- function(x,name) write.csv(x,file.path(out,paste0(name,".csv")),row.names=FALSE)
fit_mean <- function(y,states,high=NULL) {
  f <- if(is.null(high)) lm(y~1) else lm(y~high)
  k <- if(is.null(high)) 1 else 2
  if(length(coef(f))<k || anyNA(coef(f))) return(data.frame(estimate=NA,se=NA,lo=NA,hi=NA,p=NA))
  v <- vcovCL(f,cluster=states,type="HC1"); se <- sqrt(v[k,k]); b <- unname(coef(f)[k]);df <- length(unique(states))-1
  data.frame(estimate=b,se=se,lo=b-qt(.975,df)*se,hi=b+qt(.975,df)*se,p=2*pt(-abs(b/se),df))
}
averages <- list(); predictions <- list(); validation <- list(); support <- list(); folds <- list()
for(config in unique(dat$config)) {
  d <- dat[dat$config==config,]; n <- nrow(d); X <- as.matrix(d[features])
  set.seed(48123)
  states <- sample(sort(unique(d$state)))
  ntest <- max(3,floor(.4*length(states)))
  teststates <- states[seq_len(ntest)]
  train <- which(!d$state %in% teststates);test <- which(d$state %in% teststates)
  folds[[config]] <- data.frame(config=config,state=states,fold=ifelse(states %in% teststates,"test","train"))
  for(outcome in outcomes) {
    yR <- d[[paste0(outcome,"_delta_R")]]; yD <- d[[paste0(outcome,"_delta_D")]]
    fit <- function(idx) {
      yy <- c(yD[idx],yR[idx]);mid <- rep((yD[idx]+yR[idx])/2,2)
      causal_forest(X[rep(idx,2),,drop=FALSE],yy,rep(0:1,each=length(idx)),
        Y.hat=mid,W.hat=rep(.5,length(yy)),clusters=as.integer(factor(rep(d$state[idx],2))),
        honesty=TRUE,num.trees=4000,min.node.size=4,sample.fraction=.5,
        seed=72311,num.threads=2)
    }
    full <- fit(seq_len(n));full_oob <- as.numeric(predict(full)$predictions)
    oob <- (full_oob[seq_len(n)]+full_oob[n+seq_len(n)])/2
    stopifnot(all(is.finite(oob)))
    mu <- (yR+yD)/2
    augmented <- oob+(yR-(mu+oob/2))-(yD-(mu-oob/2))
    stopifnot(max(abs(augmented-(yR-yD)))<1e-9)
    for(version in c("delta","trend_adjusted","exclude2022","placebo")) {
      val <- d[[paste0(outcome,"_",version)]]
      averages[[length(averages)+1]] <- cbind(data.frame(config=config,outcome=outcome,version=version,
        pairs=n,states=length(states),forest_oob_mean=mean(oob)),fit_mean(val,d$state))
    }
    # Very small sensitivity sample still gets average contrasts, not a forced forest validation.
    if(n<25 || length(unique(d$state[train]))<8 || length(test)<8) {
      predictions[[length(predictions)+1]] <- data.frame(config=config,outcome=outcome,pair=d$pair,state=d$state,
        forest_oob=oob,fold=ifelse(seq_len(n) %in% train,"train","test"),heldout_score=NA,high=NA,
        observed_did=yR-yD,placebo=d[[paste0(outcome,"_placebo")]])
      support[[length(support)+1]] <- data.frame(config=config,outcome=outcome,train_pairs=length(train),test_pairs=length(test),
        train_states=length(unique(d$state[train])),test_states=length(teststates),high_pairs=NA,low_pairs=NA,
        high_states=NA,low_states=NA,status="Too little support for held-out forest validation")
      next
    }
    outer <- fit(train)
    score <- as.numeric(predict(outer,X[test,,drop=FALSE])$predictions)
    train_oob <- as.numeric(predict(outer)$predictions)
    cutoff <- median(train_oob)
    high <- as.numeric(score>cutoff)
    allscore <- rep(NA_real_,n);allhigh <- rep(NA_real_,n);allscore[test] <- score;allhigh[test] <- high
    predictions[[length(predictions)+1]] <- data.frame(config=config,outcome=outcome,pair=d$pair,state=d$state,
      forest_oob=oob,fold=ifelse(seq_len(n) %in% train,"train","test"),heldout_score=allscore,high=allhigh,
      observed_did=yR-yD,placebo=d[[paste0(outcome,"_placebo")]])
    ok <- length(unique(high))==2 && min(table(high))>=4 &&
      min(sapply(0:1,function(k)length(unique(d$state[test][high==k]))))>=2
    support[[length(support)+1]] <- data.frame(config=config,outcome=outcome,train_pairs=length(train),test_pairs=length(test),
      train_states=length(unique(d$state[train])),test_states=length(teststates),high_pairs=sum(high),low_pairs=sum(1-high),
      high_states=length(unique(d$state[test][high==1])),low_states=length(unique(d$state[test][high==0])),
      status=if(ok)"Exploratory; few held-out states" else "Insufficient high/low subgroup support")
    if(ok) {
      for(version in c("delta","placebo","trend_adjusted")) {
        val <- d[[paste0(outcome,"_",version)]][test]
        validation[[length(validation)+1]] <- cbind(data.frame(config=config,outcome=outcome,test=version,
          high_mean=mean(val[high==1]),low_mean=mean(val[high==0]),heldout_pairs=length(test),
          heldout_states=length(teststates)),fit_mean(val,d$state[test],high))
      }
      baseline_high <- as.numeric(X[test,1]>median(X[train,1]))
      if(length(unique(baseline_high))==2 && min(table(baseline_high))>=4)
        validation[[length(validation)+1]] <- cbind(data.frame(config=config,outcome=outcome,test="baseline_capacity_rule",
          high_mean=mean((yR-yD)[test][baseline_high==1]),low_mean=mean((yR-yD)[test][baseline_high==0]),
          heldout_pairs=length(test),heldout_states=length(teststates)),fit_mean((yR-yD)[test],d$state[test],baseline_high))
    }
  }
}
save_csv(do.call(rbind,averages),"paired_estimates")
save_csv(do.call(rbind,predictions),"forest_predictions")
save_csv(do.call(rbind,support),"forest_support")
save_csv(do.call(rbind,folds),"state_folds")
v <- if(length(validation)) do.call(rbind,validation) else data.frame(config=character(),outcome=character(),test=character(),estimate=numeric(),p=numeric())
v$p_holm_primary_family <- NA_real_
sel <- v$config=="primary" & v$test=="delta"
v$p_holm_primary_family[sel] <- p.adjust(v$p[sel],method="holm",n=3)
save_csv(v,"heldout_validation")
writeLines(capture.output(sessionInfo()),file.path(out,"R_session.txt"))
print(do.call(rbind,support))
print(v)
