.libPaths(c(Sys.getenv("TAKEOVER_R_LIB","/private/tmp/clean-grf-r46"),.libPaths()))
library(grf)
out<-commandArgs(trailingOnly=TRUE)[1]
d<-read.csv(file.path(out,"evaluation_scores.csv"),stringsAsFactors=FALSE)
areas<-read.csv(file.path(out,"areas.csv"),stringsAsFactors=FALSE)
rows<-list()
for(o in unique(d$outcome)) {
  g<-d[d$outcome==o&d$version=="delta",]
  state<-areas$state[match(g$unit,areas$unit)]
  set.seed(82317)
  rate<-rank_average_treatment_effect.fit(g$score,g$priority,target="QINI",
    q=seq(.01,1,.01),R=999,sample.weights=g$tilt,clusters=state)
  rows[[o]]<-data.frame(outcome=o,estimate=unname(rate$estimate),se=unname(rate$std.err),
    note="Native discrete-grid QINI diagnostic; SE holds estimated scores/tilts fixed")
}
write.csv(do.call(rbind,rows),file.path(out,"native_rate_diagnostic.csv"),row.names=FALSE)
