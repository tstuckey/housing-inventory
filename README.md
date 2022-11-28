#  Housing Inventory Analysis

**Tom Stuckey**  

Dec 2021  

# 1 Introduction

As the United States begins to rebound from the COVID-19 pandemic, there are many interesting happenings at both the macro-economic level and the micro-economic level that economists and econometrists will be likely researching for decades to come. On the micro-economics side, housing is interesting topic. Rocket mortgage explicitly identifies low-mortgage rates, additional remote work opportunities, and a lack of new construction.[1] At time of this writing in late 2021, most metropolitan areas have been experiencing major increases in housing prices in a decidedly strong seller's market. This analysis aims to explain housing inventory as a function of several variables.

# 2 Objective

Housing inventory availability, as shelter, is at the foundation of Maslow's hierarchy of needs.[2] As a foundational element of existence, it has many second and third order effects across the economy. While Rocket Mortgage identified some main drivers for the lack of housing inventory, there are many other variables involved. This analysis focuses on combining  several data sources from 2016 to 2020 to develop a linear model that explains overall housing inventory.


# 3 Approach

This analysis leverages the five-step framework for the data science process as graphically expressed below:
![Data Science Process](images/Data%20Science%20Process.png)
[3]

First, the interesting question is proposed in the **Ask** phase. Second, in the **Get* phase, the data is obtained and cleaned. Third, the data is explored in the **Explore** phase. Fourth, the data is modeled in the **Model** phase. Finally, the overall results are most-often encoded in a visualization and ultimately communicated in the **Communicate** phase. Each of these steps are briefly described herein and references are provided to specific notebooks with the in-depth technical analysis. 

## 3.1 Ask

The Ask phase is (or should be the) first phase of any data science process. This is where the primary question is posed and the scientific goals stated. For this analysis, the goal is to explain housing inventory as a function of several variables:

- Number of Building Permits
- Mortgage Rates
- Prime Interest Rates 
- Millions of Dollars in Revolving Credit

More detailed information on the scope of the question and the data sources is available in the notebook [Housing Inventory - Ask](Housing%20Inventory%20-%20Ask.ipynb).

## 3.2 Get 

In the Get phase, of this analysis, the data is pulled from the various repositories, cleaned, and staged in a sqlite database. This is largely achieved through a family of Python scripts. The scripts and overall process employed is described in [Housing Inventory - Get](Housing%20Inventory%20-%20Get.ipynb). 

## 3.3 Explore 

During the Explore phase, the data is first visualized in a number of different ways, and relationships are examined. This where the exploratory data analysis (EDA) should occur and does occur in this analysis. This analysis can be found in [Housing Inventory - Explore](Housing%20Inventory%20-%20Explore.ipynb).

## 3.4 Model  

The modeling phase is where the various models are build, refined, validated, and used to make predictions. For this analysis this is where the linear model is constructed and optimized. The technical details are found in [Housing Inventory - Model](Housing%20Inventory%20-%20Model.ipynb). 

## 3.5 Communicate

In the Communicate phase, the overall results are conveyed through a combination of tables and visualizations. In this analysis, the overall process is briefly revisited for holistic context in addition to the overall results. There are two elements to the Communicate phase for this analysis. [Housing Inventory Out-Brief](Housing%20Inventory%20-%20Out-Brief.pdf) contains the out-brief slides, and [YouTube](https://youtu.be/SUOOavVuJk0) provides a narrated version of the results of the housing inventory analysis. 

# 4 Conclusion  

This README overview is just the high-level overview of the housing inventory analysis effort. As mentioned above, please find the bundled notebook aligned to each particular area of the analysis for technical specfics, or, alternatively, pleas refernce the final report for an overview of the analysis in its entirety along with the conclusion.

---

**References**

[1] Ayers, C. (2021, August 13). How A Low Housing Inventory Impacts The Real Estate Market. Rocket Mortgage. Retrieved October 15, 2021, from https://www.rocketmortgage.com/learn/low-housing-inventory

[2] Mcleod, S. (2020, December 29). Maslow’s Hierarchy of Needs. Simply Psychology. Retrieved November 24, 2021, from https://www.simplypsychology.org/maslow.html

[3] Pfister, H. P., Blitzstein, J. B., & Kaynig, V. K. (2015, December 5). CS109 Data Science. Https://Github.Com/Cs109/. Retrieved November 24, 2021, from https://github.com/cs109/2015/blob/master/Lectures/01-Introduction.pdf