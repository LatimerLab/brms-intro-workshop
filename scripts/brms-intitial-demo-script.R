#### Script to work through how to use brms to fit a straightforward linear model ####
# 
# Workshop for lab meeting January 20, 2026
# 
# 1) compare linear mixed model using lme4 vs brms 
# 2) discuss the main brms settings and reason for them 
# 3) checking models for convergence 
# 4) some other model checks
# 5) a few model results plots 
# 6) posterior predictive checks
# 7) model comparison and evaluation


# A few resources: 
# Paul Bürkner's list of vignettes: https://paulbuerkner.com/brms/articles/ 
# Visualization tools: https://mjskay.github.io/tidybayes/articles/tidy-brms.html




#### Setup ####
library(brms)
library(tidyverse)
library(lme4)
library(gamlss.data) # for example data 

# Setting options to help Stan run more smoothly
# To find out how many cores a mac has, one option is to open the teminal 
#    and type: system_profiler SPHardwareDataType
rstan_options(auto_write = TRUE)
options(mc.cores = 4)


#### 1) Fit example model using lme4 

# Quick look at data 
head(rent99)
pairs(rent99[,c("rentsqm", "area", "yearc")])
ggplot(data = rent99, aes(x = district, y = rentsqm, color = district)) +
            geom_violin()

# Response variable distribution
hist(rent99$rentsqm) 


# To make this demo run faster, let's filter out a bunch of districts! 
d = rent99 |> 
     group_by(district) |> 
     summarise(n_by_district = n()) |> 
     filter(n_by_district >= 20, (district %% 2) == 0) |> 
     left_join(rent99, by = "district")

nrow(d) # ok 

# Standardize the continuous explanatory variables 


# Fit a linear mixed model with two covariates in lme4

m_lmer = lmer(rentsqm ~ area + yearc + (1 | district), data = d)

summary(m_lmer)
plot(m_lmer)
qqnorm(resid(m_lmer))


#### 2) Fit the same model using brms 

# Formula (usually the same as above for simpler models)

# Family (usually the same as above for simpler models)

# Prior 

# Control 






