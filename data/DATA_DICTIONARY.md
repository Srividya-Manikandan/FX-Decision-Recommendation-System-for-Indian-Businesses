# FX Decision Support System - Data Dictionary

This document defines the attributes and metrics used in the preprocessed foreign exchange dataset (`cleaned_fx_data.csv`). This data serves as the foundation for the Risk Scoring Engine and Forecasting models.

## 1. Core Identifiers
| Attribute | Data Type | Description |
| :--- | :--- | :--- |
| **Date** | Date (Index) | The daily timestamp for the exchange rate. Includes weekends and holidays (imputed). |

## 2. Exchange Rates (Base: INR)
*Source: RBI (Pre-2018) & FBIL (Post-2018)*
| Attribute | Data Type | Description |
| :--- | :--- | :--- |
| **USD** | Float | Exchange rate for 1 US Dollar in Indian Rupees. |
| **GBP** | Float | Exchange rate for 1 British Pound Sterling in Indian Rupees. |
| **EUR** | Float | Exchange rate for 1 Euro in Indian Rupees. |
| **JPY** | Float | Exchange rate for 100 Japanese Yen in Indian Rupees. |

## 3. Derived Analytical Metrics
*Calculated during the Preprocessing Phase*
| Attribute | Formula | Description |
| :--- | :--- | :--- |
| **[CUR]_Return** | `(Price_t / Price_t-1) - 1` | The daily percentage change in the currency value. Used for stationarity in forecasting. |
| **[CUR]_Volatility** | `Rolling_Std(Returns, 30)` | 30-day moving standard deviation of returns. Represents market "nervousness" or risk level. |

## 4. Data Processing Notes
* **Gap Filling:** Weekends and bank holidays were filled using the **Forward Fill (ffill)** method, carrying the last known market-close price forward.
* **Merging:** Data from 2016-2018 (RBI) and 2018-2026 (FBIL) was merged using a `combine_first` strategy to ensure a continuous 10-year timeline.
* **Outliers:** Daily returns exceeding +/- 3 standard deviations are flagged for review as "Market Shock" events.