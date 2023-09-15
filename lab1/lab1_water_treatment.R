setwd("C:/Users/nilan/OneDrive - Rensselaer Polytechnic Institute/RPI/My Acadamic/Data Analaytics/DataAnalytics2023_Neelanga_Thelasingha/lab1/w_treat/")
wtrt_data<-read.csv("water_treatment.csv",header=T)
#GMP_data<-read.csv("GPW3_GRUMP_SummaryInformation_2010.csv",header=T,skip = 1)
#or
#EPI_data <- read.xlsx(”<path>/2010EPI_data.xlsx")
# Note: replace default data frame name – cannot start with numbers!
View(wtrt_data)
#
#detach(EPI_data)
attach(wtrt_data) 	# sets the ‘default’ object
fix(wtrt_data) 	# launches a simple data editor

PH.E			# prints out values
tf <- is.na(PH.E) # records True values if the value is NA
PHE <- PH.E[!tf] # filters out NA values, new array

#other data
#GRUMP_data <- read.csv(”<path>/GPW3_GRUMP_SummaryInformation_2010.csv")

View(PHE)
summary(PHE) 	# stats
fivenum(PH.E,na.rm=TRUE)
help(stem)
stem(PH.E)		 # stem and leaf plot
help(hist)
hist(PH.E)
hist(PH.E, seq(6.5, 9, 0.1), prob=TRUE)
help(lines)
lines(density(PH.E,na.rm=TRUE,bw="SJ")) # or try bw=“SJ”
help(rug)
rug(PH.E) 



#Cumulative Density Function
plot(ecdf(PH.E), do.points=FALSE, verticals=TRUE) 
#Quantile-Quantile?
par(pty="s") 
qqnorm(PH.E); qqline(PH.E)
#Simulated data from t-distribution:
x <- rt(250, df = 5)
qqnorm(x); qqline(x)
#Make a Q-Q plot against the generating distribution by: x<-seq(30,95,1)
qqplot(qt(ppoints(250), df = 5), x, xlab = "Q-Q plot for t dsn")
qqline(x)

SED<-as.numeric(SED.E)
COND<-as.numeric(COND.E)
boxplot(SED,COND)
boxplot(COND)
boxplot(SED)
qqplot(SED,COND)


shapiro.test(SED)
shapiro.test(COND)
summary(SED)
fivenum(COND,na.rm=TRUE)




