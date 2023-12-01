#########################################################################################################################
###
### R code to perform survival analysis on players drafted or signed by NBA teams
### Author: Sean Farrell
### Last update: December 1st 2023
###
#########################################################################################################################

# Load libraries 
library(survival)
library(pbapply)
library(data.table)
library(caret)
library(pbmcapply)
library(ggplot2)
library(plyr)
library(dplyr)
library(ggfortify)
library(gtsummary)


# Load the training set
train <- read.csv("training_set.csv")

# Discarded athletes who don't get signed
train <- train[train$LABEL == "Y",]

# Drop low base rate features, word count and any related to punctuation
d <- lapply(7:ncol(train),function(i){
  m1 <- median(train[,i])
  m2 <- mean(train[,i])
  
  output <- data.frame(Feature=names(train)[i],Mean=m2,Median=m1)
  return(output)
})
temp <- rbindlist(d)

drop_vars <- c("WC","WPS","AllPunc","Period","Comma","QMark","Exclam","Apostro","OtherP","Emoji",temp$Feature)
drop_vars # Ignore these variables in the CPH models

# Group athletes by draft position or status
train$Round <- "Not Drafted"
train$Round[train$OVERALL_PICK <= 10] <- "Early 1st Round"
train$Round[train$OVERALL_PICK > 10 & train$OVERALL_PICK <= 20] <- "Mid 1st Round"
train$Round[train$OVERALL_PICK > 20 & train$OVERALL_PICK <= 30] <- "Late 1st Round"
train$Round[train$OVERALL_PICK > 30 & train$OVERALL_PICK <= 40] <- "Early 2nd Round"
train$Round[train$OVERALL_PICK > 40 & train$OVERALL_PICK <= 50] <- "Mid 2nd Round"
train$Round[train$OVERALL_PICK > 50 & train$OVERALL_PICK <= 60] <- "Late 2nd Round"
train$Round <- factor(train$Round,levels=c("Early 1st Round", "Mid 1st Round","Late 1st Round",
                                           "Early 2nd Round", "Mid 2nd Round","Late 2nd Round",
                                           "Not Drafted"))

# Identify censored events (i.e. playing career "death")
train$Censored <- 0
train$Censored[train$PlayingStatus == ""] <- 1

# Drop unnecessary columns (including punctuation features)
train <- subset(train,select=-c(id,OVERALL_PICK,PlayingStatus,LABEL))

### First, let's generate Kaplan-Meier curves just using the draft position columns
km_trt_fit <-survfit(Surv(Duration,Censored) ~ Round, data=train)
autoplot(km_trt_fit,xlab = "Career Duration (Years)",ylab="Career Survival Probability") + theme_bw() + 
  labs(fill="Draft Status",color="Draft Status") 


# One-hot encode draft status columns for Cox proportional hazard modelling
train$Round1.1 <- 0
train$Round1.2 <- 0
train$Round1.3 <- 0
train$Round2.1 <- 0
train$Round2.2 <- 0
train$Round2.3 <- 0
train$NoDraft <- 0

train$Round1.1[train$Round == "Early 1st Round"] <- 1
train$Round1.2[train$Round == "Mid 1st Round"] <- 1
train$Round1.3[train$Round == "Late 1st Round"] <- 1
train$Round2.1[train$Round == "Early 2nd Round"] <- 1
train$Round2.2[train$Round == "Mid 2nd Round"] <- 1
train$Round2.3[train$Round == "Late 2nd Round"] <- 1
train$NoDraft[train$Round == "Not Drafted"] <- 1

### Build Cox proportional hazard regression model using just draft status columns
cox1 <- coxph(Surv(Duration,Censored) ~ Round1.1 + Round1.2 + Round1.3 + Round2.1 + Round2.2 + Round2.3 + NoDraft,
             data = train)
summary(cox1)

# Concordance= 0.778  (se = 0.009 )
# Likelihood ratio test= 369.1  on 6 df,   p=<2e-16
# Wald test            = 364.2  on 6 df,   p=<2e-16
# Score (logrank) test = 453.3  on 6 df,   p=<2e-16


### Build Cox proportional hazard regression model using just LIWC features
cox2 <- coxph(Surv(Duration,Censored) ~ Analytic + Clout + Authentic + Tone + BigWords + Dic + Linguistic + function. + pronoun + ppron + i + we + you + shehe + they + ipron + det + article + number + prep + auxverb + adverb + conj + negate + verb + adj + quantity + Drives + affiliation + achieve + power + Cognition + allnone + cogproc + insight + cause + discrep + tentat + certitude + differ + Affect + tone_pos + tone_neg + emotion + emo_pos + Social + socbehav + prosocial + comm + socrefs + male + Lifestyle + leisure + work + need + acquire + allure + Perception + motion + space + visual + feeling + time + focuspast + focuspresent + focusfuture ,
             data = train)
summary(cox2)

# Concordance= 0.617  (se = 0.014 )
# Likelihood ratio test= 95.24  on 66 df,   p=0.01
# Wald test            = 92.8  on 66 df,   p=0.02
# Score (logrank) test = 92.87  on 66 df,   p=0.02

# Finally, build Cox proportional hazard model with LIWC + draft status columns
cox3 <- coxph(Surv(Duration,Censored) ~ Analytic + Clout + Authentic + Tone + BigWords + Dic + Linguistic + function. + pronoun + ppron + i + we + you + shehe + they + ipron + det + article + number + prep + auxverb + adverb + conj + negate + verb + adj + quantity + Drives + affiliation + achieve + power + Cognition + allnone + cogproc + insight + cause + discrep + tentat + certitude + differ + Affect + tone_pos + tone_neg + emotion + emo_pos + Social + socbehav + prosocial + comm + socrefs + male + Lifestyle + leisure + work + need + acquire + allure + Perception + motion + space + visual + feeling + time + focuspast + focuspresent + focusfuture +
               Round1.1 + Round1.2 + Round1.3 + Round2.1 + Round2.2 + Round2.3 + NoDraft,
             data = train)
summary(cox3)

# Concordance= 0.803  (se = 0.009 )
# Likelihood ratio test= 478  on 72 df,   p=<2e-16
# Wald test            = 443.1  on 72 df,   p=<2e-16
# Score (logrank) test = 549.5  on 72 df,   p=<2e-16


# Generate bar chart of different concordance values
temp <- rbind(as.data.frame(t(cox2$concordance)),as.data.frame(t(cox1$concordance)),as.data.frame(t(cox3$concordance)))
temp <- cbind(Features=c("LIWC","Draft Status","LIWC + Draft Status"),temp)
temp$Features <- factor(temp$Features,levels=rev(temp$Features))

ggplot(temp, aes(x = concordance, y = Features,fill=Features)) +
  geom_bar(stat = "identity", color = "black") +
  geom_errorbar(aes(xmin = concordance - std, xmax = concordance + std)) +
  labs(x = "Concordance",
       y = "Model") +
  geom_vline(xintercept = 0.5,linetype="dashed") +
  theme_bw() + theme(axis.text.y = element_text(angle = 45, hjust = 1),legend.position = "none") +
  scale_fill_manual(values=rev(c("#66C2A5", "#3288BD", "#5E4FA2"))) +
  # scale_fill_manual(values=c("dodgerblue4","steelblue3","steelblue1")) +
  scale_x_continuous(breaks = seq(0, 1, by = 0.1))


# Create survival curves for LIWC only Cox model
cox_fit <- survfit(cox2)

# Generate hazard ratios for all features
p <- summary(cox2)$coefficients[, "Pr(>|z|)"]
co <- summary(cox2)$coefficients[, "exp(coef)"]
co <- co[order(co)]
cox2 %>% tbl_regression(include=names(co),exp=T) %>%  bold_p(t = 0.05)

# Generate hazard ratios only for features with p < 0.05
p <- summary(cox2)$coefficients[, "Pr(>|z|)"]
co <- summary(cox2)$coefficients[, "exp(coef)"]
co <- co[order(co)]
idx <- which(p < 0.05)
p <- p[idx]
co <- co[names(co) %in% names(p)]
cox2 %>% tbl_regression(include=names(co),exp=T) %>%  bold_p(t = 0.001)


### Plot final survival curve for LIWC only Cox model
autoplot(cox_fit,xlab = "Career Duration (Years)",ylab="Career Survival Probability",surv.colour = 'blue',censor.colour = 'red',censor.size=6) + theme_bw() + 
  labs(fill="Draft Status",color="Draft Status") + geom_vline(xintercept = 4,linetype="dashed")


### The Drives hazard ratio is quite high, so lets look at how adding risk and reward back in change things
cox4 <- coxph(Surv(Duration,Censored) ~ Analytic + Clout + Authentic + Tone + BigWords + Dic + Linguistic + function. + pronoun + ppron + i + we + you + shehe + they + ipron + det + article + number + prep + auxverb + adverb + conj + negate + verb + adj + quantity + Drives + affiliation + achieve + power + Cognition + allnone + cogproc + insight + cause + discrep + tentat + certitude + differ + Affect + tone_pos + tone_neg + emotion + emo_pos + Social + socbehav + prosocial + comm + socrefs + male + Lifestyle + leisure + work + need + acquire + allure + Perception + motion + space + visual + feeling + time + focuspast + focuspresent + focusfuture + risk + reward,
              data = train)
summary(cox4)

# Drives hazard ratios
# Drives = 1.483, p = 0.04
# risk = 0.859, p = 0.54
# reward = 1.53, p = 0.72
# affiliation = 0.676, p = 0.08
# achieve = 0.709, p = 0.07
# power = 0.7344, p = 0.09

### Try again but drop Drives
cox5 <- coxph(Surv(Duration,Censored) ~ Analytic + Clout + Authentic + Tone + BigWords + Dic + Linguistic + function. + pronoun + ppron + i + we + you + shehe + they + ipron + det + article + number + prep + auxverb + adverb + conj + negate + verb + adj + quantity + affiliation + achieve + power + Cognition + allnone + cogproc + insight + cause + discrep + tentat + certitude + differ + Affect + tone_pos + tone_neg + emotion + emo_pos + Social + socbehav + prosocial + comm + socrefs + male + Lifestyle + leisure + work + need + acquire + allure + Perception + motion + space + visual + feeling + time + focuspast + focuspresent + focusfuture + risk + reward,
              data = train)
summary(cox5)

# Drives hazard ratios
# risk = 0.883, p = 0.62
# reward = 1.014, p = 0.92
# affiliation = 0.981, p = 0.88
# achieve = 1.025, p = 0.70
# power = 1.037, p = 0.62