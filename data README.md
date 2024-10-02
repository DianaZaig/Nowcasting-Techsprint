The basic idea for data gathering was to select various data sources that are considered to have potential informational value regarding liquidity risk for individual banks. 
Therefore, we went through an extensive data selection and evaluation phase. In this step of the TechSprint, we considered the following potential data sources:

Regulatory data (e.g., LCR)
Financial data from the credit institution (e.g., P&L)
Market data (e.g., CDS)
Macro data (e.g., GDP)
Social media data (e.g., Tweets)

Based on expert judgment, all of these data sources could potentially be helpful for the task of nowcasting liquidity risk metrics. 
However, further testing should be done to find an optimal database for training.

To quickly develop the model and have a working prototype by the end of the event, the final data selection was based on 
existing research (https://www.sciencedirect.com/science/article/pii/S2666827023000646?via%3Dihub#sec2) on the use of Machine Learning (ML) to predict liquidity risk. 
The target variable in this setting was:

MLA = [Total liquid assets / Demand or liquid liabilities] × 100%

The input variables are as follows:

1. **Current Ratio (X1)**: Measures the bank's ability to meet short-term obligations. Formula: \[Current assets / Current liabilities\].

2. **Liquid Assets to Deposit (X2)**: Assesses the funding of liquid assets by deposits. Formula: \[Liquid Assets / Deposit\].

3. **Deposit (X3)**: Indicates the level of deposits, which increases with liquidity risk.

4. **Core Deposit (X4)**: Evaluates the stability of deposits. A poor ratio of \[X4 / X3\] indicates over-reliance on unstable non-core deposits.

5. **Loan to Deposit (X5)**: Determines the funding of loans by deposits. Formula: \[Loan / Deposit\]. A ratio above 80% increases exposure to credit risk.

6. **Volatile Deposits to Total Liabilities (X6)**: Measures the proportion of volatile deposits. Formula: \[Volatile Deposits / Total Liabilities\].

7. **Credit in the Central Bank to Total Deposits (X7)**: Reflects the liquidity position. Formula: \[Credit in the Central Bank / Total Deposits\].

8. **Net Loans and Leases to Total Assets (X8)**: Assesses the funding of net loans and leases by total assets. Formula: \[Net Loans and Leases / Total Assets\].

9. **Net Loans and Leases to Deposits (X9)**: Evaluates the funding of net loans and leases by deposits. Formula: \[Net Loans and Leases / Deposits\].

10. **Net Loans and Leases to Core Deposits (X10)**: Determines the funding of net loans and leases by core deposits. Formula: \[Net Loans and Leases / Core Deposits\].

11. **Demand Deposits to Total Assets (X11)**: Measures the funding of demand deposits by total assets. Formula: \[Demand Deposits / Total Assets\].

12. **Bank Size (X12)**: Logarithm of total assets. Formula: \[Log10(Total Assets)\].

13. **Inflation (X13)**: High inflation decreases liquidity risk by increasing demand for loans and collateral values.

14. **GDP Growth (X14)**: High GDP growth increases liquidity risk by boosting economic activities and loan demand.

15. **Loan to Assets (X15)**: Determines the percentage of asset-funded loans. Formula: \[Loan / Assets\].

16. **Non-Performing Loans (NPL) (X16)**: Measures the proportion of non-performing loans. Formula: \[Non-Performing Loans / Total Outstanding Loans × 100%\].

17. **NPL Net of Provisions (NPLNP) (X17)**: Better measure of non-performing loans after provisions. Formula: \[(Non-Performing Loans – Provisions) / Total Outstanding Loans × 100%\].

18. **NPLNP to Core Capital (NPLNPCC) (X18)**: Indicates the ability of core capital to absorb NPL losses. Formula: \[NPLNP / Core Capital × 100%\].

19. **Net Interest Margin (NIM) (X19)**: Measures income from loans and investments. Formula: \[Interest Earned / Interest Paid\].

20. **Lending Rate (LeR) (X20)**: Higher lending rates negatively affect liquidity risk by decreasing loan demand and increasing funding costs.

21. **Core Deposit to Total Assets (CDTA) (X21)**: Determines the concentration of stable liquidity-providing assets. Formula: \[Core Deposit / Total Assets\].

22. **Liquid Assets to Total Assets (LATA) (X22)**: Assesses the bank's ability to withstand liquidity shocks. Formula: \[Liquid Assets / Total Assets\].

23. **Excess Short-Term Liabilities to Total Assets (X23)**: Evaluates the incorrect funding of long-term assets by short-term liabilities.

24. **Significant Credit Risk Event (X24)**: Measures the impact of credit deterioration on liquidity.

25. **Operational Risk Loss Event (X25)**: Assesses operating losses and risks that reduce liquidity.

26. **Return On Equity (ROE) (X26)**: Determines liquidity using earning trends. Formula: \[Net Income / Shareholders’ Equity\].

27. **Capital Adequacy Ratio (CAR) (X27)**: Demonstrates the resilience of the bank’s capital. Formula: \[(Tier I + Tier II + Tier III Capital) / Total Risk-Weighted Assets\].

28. **Domestic Credit (X28)**: Measures the effect of market conditions on liquidity risk.

29. **Market Rumours about the Bank (X29)**: Assesses the impact of market conditions on liquidity risk.

30. **Return on Asset (ROA) (X30)**: Illustrates how a bank makes money from its assets. Formula: \[Net Profit Before Tax / Average of Total Assets\].

