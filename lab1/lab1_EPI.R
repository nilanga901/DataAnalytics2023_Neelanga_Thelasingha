setwd("C:/Users/nilan/OneDrive - Rensselaer Polytechnic Institute/RPI/My Acadamic/Data Analaytics/DataAnalytics2023_Neelanga_Thelasingha/lab1/EPI/")
#EPI_data<-read.csv("2010EPI_data.csv",header=T)
EPI_data<-read.csv("2010EPI_data.csv",header=T,skip = 1)
#or
#EPI_data <- read.xlsx(”<path>/2010EPI_data.xlsx")
# Note: replace default data frame name – cannot start with numbers!
View(EPI_data)
#
attach(EPI_data) 	# sets the ‘default’ object
fix(EPI_data) 	# launches a simple data editor
EPI 			# prints out values EPI_data$EPI
tf <- is.na(EPI) # records True values if the value is NA
E <- EPI[!tf] # filters out NA values, new array

#other data
#GRUMP_data <- read.csv(”<path>/GPW3_GRUMP_SummaryInformation_2010.csv")

View(E)
summary(EPI) 	# stats
fivenum(EPI,na.rm=TRUE)
help(stem)
stem(EPI)		 # stem and leaf plot
help(hist)
hist(EPI)
hist(EPI, seq(30., 95., 1.0), prob=TRUE)
help(lines)
lines(density(EPI,na.rm=TRUE,bw=1.)) # or try bw=“SJ”
help(rug)
rug(EPI) 


#Cumulative Density Function
plot(ecdf(EPI), do.points=FALSE, verticals=TRUE) 
#Quantile-Quantile?
par(pty="s") 
qqnorm(EPI); qqline(EPI)
#Simulated data from t-distribution:
x <- rt(250, df = 5)
qqnorm(x); qqline(x)
#Make a Q-Q plot against the generating distribution by: x<-seq(30,95,1)
qqplot(qt(ppoints(250), df = 5), x, xlab = "Q-Q plot for t dsn")
qqline(x)

boxplot(EPI,DALY)
qqplot(EPI,DALY)

boxplot(EPI,ENVHEALTH, ECOSYSTEM, DALY, AIR_H,WATER_H, AIR_E,WATER_E,BIODIVERSITY)
qqplot(EPI,ENVHEALTH)
qqplot(EPI, ECOSYSTEM)
qqplot(EPI, AIR_H)
qqplot(EPI,WATER_H)
qqplot(EPI, AIR_E)
qqplot(EPI,WATER_E)
qqplot(EPI,BIODIVERSITY)



help(distributions)
# try different ones.....
#Landlock
EPILand<-EPI[!Landlock]
Eland <- EPILand[!is.na(EPILand)]
#
hist(Eland)
hist(Eland, seq(30., 95., 1.0), prob=TRUE)
shapiro.test(EPI)

shapiro.test(DALY)
summary(DALY)
fivenum(DALY,na.rm=TRUE)
