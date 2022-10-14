# G2M Case Study



## Background –G2M(cab industry) case study
- XYZ is a private equity firm in US. Due to remarkable growth in the Cab Industry in last few years and multiple key players in the market, it is planning for an investment in Cab industry. 

- Objective:
 Provide actionable insights to help XYZ firm in identifying the right company for making investment.

- The analysis has been divided into four parts: 
 1. Data Understanding 
 2. Forecasting profit and number of rides for each cab type 
 3. Finding the most profitable Cab company 
 4. Recommendations for investment
---
## Data Exploration

- 24 Features( including 9 derived features)
- Timeframe of the data: 2016-01-31 to 2018-12-31
- Total data points :355,032


Assumptions:

- Outliers are present in Price_Charged feature but due to 
      unavailability of trip duration details ,we are not treating this as outlier.

- Profit of rides are calculated keeping other factors constant and only 
      Price_Charged and Cost_of_Trip features used to calculate profit.

- Users feature of city dataset is treated as number of cab users in the city.
      we have assumed that this can be other cab users as well(including Yellow and
      Pink cab) 

---

