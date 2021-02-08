CIR=read.table("CIR.csv",sep=",",quote="",header=T)
CDR=read.table("CDR.csv",sep=",",quote="",header=T)
death_rate=c()
for (i in 1:nrow(CIR)){
for (j in 1:nrow(CDR)){
if ((CIR[i,3]==CDR[j,2]) && (CIR[i,4]==CDR[j,3]))
death_rate=cbind(as.matrix(CIR[i,-1]),as.matrix(CDR[j,c(4,5,7,9)]),CIR[i,11]*CDR[j,9])
death_rate=rbind(death_rate,death_rate)
}
}
colnames(death_rate)=c("T0","predict_factor","bed/thousand","M_15_percent","M_15_65_percent","M_65_percent","F_15_percent","F_15_65_percent","F_65_percent","infection_rate","ICU/thousand","M_less_65","F_less_65","death_infection_rate","death_rate_total")

write.table(unique(death_rate),"death_rate.csv",sep=",",quote=F)

