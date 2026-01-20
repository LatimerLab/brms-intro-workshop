#### Script to work through how to use brms to fit a linear mixed model ####
# 
# Workshop for lab meeting January 20, 2026
#
# Covers: 
# 1) compare linear mixed model using lme4 vs brms 
# 2) discuss the main brms settings and reason for them 
# 3) checking models for convergence 
# 4) some other model checks including posterior predictive checks
# 5) very brief intro to model comparison

# Not covered:
# - more thorough model evaluation
# - predictions and looking at distributions of predictions
# - mechanics of MCMC 
# - bayesian modeling in general 

# A few resources: 
# Paul Bürkner's list of vignettes: https://paulbuerkner.com/brms/articles/ 
# Visualization tools: https://mjskay.github.io/tidybayes/articles/tidy-brms.html
# General comments on priors:  https://github.com/stan-dev/stan/wiki/prior-choice-recommendations


#### Setup ####
library(brms)
library(rstan)
library(tidyverse)
library(lme4)
library(gamlss.data) # for example data 

# optional library for using DHARMa with model output
#remotes::install_github("Pakillo/DHARMa.helpers")
library(DHARMa)
library(DHARMa.helpers)
library(bayesplot)

# Set options to help Stan run more smoothly.
# Note: to find out how many cores a mac has, one option is to open the teminal 
#    and type: system_profiler SPHardwareDataType
rstan_options(auto_write = TRUE)
options(mc.cores = 4)


#### 1) Fit example model using lme4  ####

# Current dataset: Munich rental prices for most years 1918 to 1997
# See: https://www.rdocumentation.org/packages/gamlss.data/versions/6.0-6/topics/rent99
# Response variable: monthly rent per square meter in Euros
# Explanatory variables: yearc (year of construction), area (total area of unit)
# Grouping variable: district (which part of the city it's in)


# Quick look at data 
head(rent99)
pairs(rent99[,c("rentsqm", "area", "yearc")])

# Response variable distribution
hist(rent99$rentsqm) 


# To make this demo run faster, let's filter out a bunch of districts! 
d = rent99 |> 
        group_by(district) |> 
        summarise(n_by_district = n()) |> 
        filter(n_by_district >= 20, (district %% 2) == 0) |> 
        left_join(rent99, by = "district") |> 
        select(-location, -bath, -kitchen, -cheating, -n_by_district) 

nrow(d) # ok 
head(d)

# Standardize the continuous explanatory variables 
d = d |> 
     mutate(area_std = scale(area), yearc_std = scale(yearc))


# Fit a linear mixed model with two covariates in lme4

m_lmer = lmer(rentsqm ~ area_std + yearc_std + (1 | district), data = d)


# Residuals plots 
sim_res <- simulateResiduals(fittedModel = m_lmer, n = 1000)
plot(sim_res)

# Results 
summary(m_lmer) 


#### 2) Fit the same model using brms ####

# We use brm() to fit the model
# First we can go ahead and just use the default priors and control options

m_brm = brm(rentsqm ~ area_std + yearc_std + (1 | district), data = d)

# Note there's a bunch of stuff printed out while it runs!
# We can talk about what this all means. 

# Printout of results 
summary(m_lmer)

# Step back and take a look at the chains and other results

# The chains themselves and convergence diagnostics
mcmc_plot(m_brm, type = "trace")
mcmc_plot(m_brm, type = "acf")
mcmc_plot(m_brm, type = "rhat")

# Residuals plots 
dh_check_brms(m_brm)

# Posterior predictive check -- do simulated data from the model look similar to the real data? 
pp_check(m_brm, type = "dens_overlay")

# Results plots
mcmc_pairs(x = m_brm, pars = c("b_Intercept", "sigma", "b_area_std", "b_yearc_std", "sd_district__Intercept"))
mcmc_plot(m_brm)
conditional_effects(m_brm)




#### 3) Go through the additional settings/controls for brms ####

## A) Priors 
# These represent the information we had before learning from the new data.
# 
# Next we can look at what priors the model is using by default. 
#    and see how to choose our own. 

# Note we can always check what brms' Stan code looks like 
#    But often it's hard to interpret unless you know Stan well. 
# stancode(rentsqm ~ area_std + yearc_std + (1 | district), data = d)

get_prior(m_brm)
# This shows: 
#    flat priors on regression coefficients (class b)
#    heavy-tailed Student_t priors on standard deviation parameters, 
#         including both the sd for random intercepts for district (class sd) 
#         and the residual sd (class sigma)

# Let's set our own priors
test_prior = prior(normal(0, 3), class = "b", coef = "area_std") +
                    prior(normal(0, 3), class = "b", coef = "yearc_std") +
                    prior(cauchy(0, 25), class = "sd") + 
                    prior(cauchy(0, 25), class = "sigma")

# Rerun model with test priors 
m2_brm = m_brm = brm(rentsqm ~ area_std + yearc_std + (1 | district), 
                              data = d, prior = test_prior)

m2_brm


# B) MCMC control parameters 

# These control how stan will fit the model using MCMC. 

# Example set of control parameters 
control_params = list(adapt_delta = 0.9, # hamiltonian mcmc control parameter
                     # sometimes have to increase closer to 1
                    max_treedepth = 20 # ditto 
                     # sometimes have to increase beyond 10)

# Rerun model with these control parameters 

m3_brm = m_brm = brm(rentsqm ~ area_std + yearc_std + (1 | district), 
                    data = d, 
                    prior = test_prior, 
                    control = control_params, 
                    chains = 5, # how many MCMC chains to run
                    iter = 2000, # how many steps to run each chain
                    warmup = 1000, # how many of the early steps to discard
                                   # set this long enough that chains converged
                    cores = 5 # how many CPU units to use (usually 1 per chain)
                    # init = [can set initial values of parameters]
                                   # otherwise generated randomly
                    )

m3_brm



#### 4) Model extensions and comparison ####

## Some pretty straightforward things you can add to a model in brms 
# -- Add nonlinear terms (gam)
# -- Model the variance parameter(s) or other model terms 
# -- Add "shrinkage priors" that can do some automatic variable selection 
# -- Use pretty much any model distribution, plus deal with zero inflation, ordinal response variables, multivariate response variables. 
# -- Add spatial random effects via a Gaussian Process prior 
# -- Predict any quantity (e.g. predictions for new locations) 
#    as part of model fitting to carry uncertainty directly into predictions. 

# OF COURSE the more complex the model gets, the more likely we'll have issues.


## Very short demo of turning our model into a gam  

gam_brm = brm(rentsqm ~ s(area_std) + s(yearc_std) + (1 | district), data = d, 
              control = control_params, 
              chains = 4, # how many MCMC chains to run
              iter = 2000, # how many steps to run each chain
              warmup = 1000, # how many of the early steps to discard
              # set this long enough that chains converged
              cores = 4 # how many CPU units to use (usually 1 per chain)
              # init = [can set initial values of parameters]
              # otherwise generated randomly
              )

# How nonlinear are the effects?
marginal_effects(gam_brm)

# Residuals plots 
dh_check_brms(gam_brm) # !

pp_check(gam_brm, type = "dens_overlay")

WAIC(gam_brm, m3_brm)
loo(gam_brm, m3_brm)


# Does an interaction of the smooth terms help the fit? 
gam2_brm = brm(rentsqm ~ s(area_std, yearc_std) + (1 | district), data = d, 
              control = control_params, 
              chains = 4, # how many MCMC chains to run
              iter = 2000, # how many steps to run each chain
              warmup = 1000, # how many of the early steps to discard
              # set this long enough that chains converged
              cores = 4 # how many CPU units to use (usually 1 per chain)
              # init = [can set initial values of parameters]
              # otherwise generated randomly
              )

conditional_effects(gam2_brm)
loo(gam_brm, gam2_brm)


# Gavin Simpsons very short bit on gams in brm: https://fromthebottomoftheheap.net/2018/04/21/fitting-gams-with-brms/

