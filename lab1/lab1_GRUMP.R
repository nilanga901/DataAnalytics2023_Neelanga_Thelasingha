setwd("C:/Users/nilan/OneDrive - Rensselaer Polytechnic Institute/RPI/My Acadamic/Data Analaytics/DataAnalytics2023_Neelanga_Thelasingha/lab1/GRUMP/")
GMP_data<-read.csv("GPW3_GRUMP_SummaryInformation_2010.csv",header=T)
#GMP_data<-read.csv("GPW3_GRUMP_SummaryInformation_2010.csv",header=T,skip = 1)
#or
#EPI_data <- read.xlsx(”<path>/2010EPI_data.xlsx")
# Note: replace default data frame name – cannot start with numbers!
View(GMP_data)
#
#detach(EPI_data)
attach(GMP_data) 	# sets the ‘default’ object
fix(GMP_data) 	# launches a simple data editor
PopulationPerUnit 			# prints out values EPI_data$EPI
tf <- is.na(PopulationPerUnit) # records True values if the value is NA
PPU <- PopulationPerUnit[!tf] # filters out NA values, new array

#other data
#GRUMP_data <- read.csv(”<path>/GPW3_GRUMP_SummaryInformation_2010.csv")

View(PPU)
summary(PopulationPerUnit) 	# stats
fivenum(PopulationPerUnit,na.rm=TRUE)
help(stem)
stem(PopulationPerUnit)		 # stem and leaf plot
help(hist)
hist(PopulationPerUnit)
hist(PopulationPerUnit, seq(0., 3000., 100.0), prob=TRUE)
help(lines)
lines(density(PopulationPerUnit,na.rm=TRUE,bw=1.)) # or try bw=“SJ”
help(rug)
rug(PopulationPerUnit) 



#Cumulative Density Function
plot(ecdf(PopulationPerUnit), do.points=FALSE, verticals=TRUE) 
#Quantile-Quantile?
par(pty="s") 
qqnorm(PopulationPerUnit); qqline(PopulationPerUnit)
#Simulated data from t-distribution:
x <- rt(250, df = 5)
qqnorm(x); qqline(x)
#Make a Q-Q plot against the generating distribution by: x<-seq(30,95,1)
qqplot(qt(ppoints(250), df = 5), x, xlab = "Q-Q plot for t dsn")
qqline(x)

Area<-as.numeric(Area)
boxplot(Area,PopulationPerUnit)
boxplot(PopulationPerUnit)

boxplot(Area)

qqplot(Area,PopulationPerUnit)


shapiro.test(Area)
shapiro.test(PopulationPerUnit)
summary(Area)
fivenum(Area,na.rm=TRUE)

summary(PopulationPerUnit)
fivenum(PopulationPerUnit,na.rm=TRUE)

help(distributions)
# try different ones.....
#Landlock


