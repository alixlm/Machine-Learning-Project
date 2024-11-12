# ⚡ Machine Learning Project: Electricity Price Explanation ⚙️📉

# This project is under development

This group project aims to explain electricity prices based on meteorological, energy, and commercial data for two European countries: France 🇫🇷 and Germany 🇩🇪. The goal is to explain daily variations in electricity futures prices using variables such as temperature 🌡️, electricity consumption ⚡, commodity prices ⛽, and more.

## 🌍 Context

Electricity prices are influenced by numerous factors on a daily basis. For instance, local climate variations can affect both electricity demand and supply 🌧️🌬️. In the long run, phenomena like climate change 🌡️ have a lasting impact on prices. Geopolitical events, such as the war in Ukraine 🇺🇦, can also influence commodity costs, with each country relying on its unique energy mix (nuclear ⚛️, solar ☀️, hydro 💧, natural gas ⛽, coal 🪨, etc.).

Additionally, each country can import or export electricity through dynamic markets, especially in Europe 🇪🇺. These diverse factors make it challenging to model electricity prices for each country.

Our model uses a machine learning algorithm to explain price variations based on these data points, optimizing performance with Spearman correlation 📈.

## 🎯 Objectives

The objective is to build a model capable of providing an accurate estimate of daily changes in electricity futures prices in France and Germany. Futures contracts allow for the purchase or sale of a set amount of electricity at a fixed price, to be delivered at a future date. This project focuses on short-term futures (24h maturity) to estimate electricity prices based on current market conditions.

## 📊 Evaluation

The model’s performance is measured by the Spearman correlation between the model’s predictions and the actual variations in futures prices in the test dataset ✅.

## 📁 Data

The data is organized into two sets: `X_train` and `X_test`, containing the same explanatory variables over different time periods.

### Files and Columns

- **X_train.csv**: Training input data
- **X_test.csv**: Test input data
  - **35 columns**
  - `ID`: Unique identifier (index), associated with a day (`DAY_ID`) and a country (`COUNTRY`)
  - `DAY_ID`: Day identifier (dates have been anonymized)
  - `COUNTRY`: Country identifier (`DE` = Germany 🇩🇪, `FR` = France 🇫🇷)
  - `GAS_RET`, `COAL_RET`, `CARBON_RET`: Futures prices for natural gas, coal, and carbon emissions in Europe

  **Weather, energy production, and consumption variables** (daily values per country):
  - `x_TEMP`: Temperature 🌡️
  - `x_RAIN`: Rain 🌧️
  - `x_WIND`: Wind 🌬️
  - `x_GAS`: Natural gas ⛽
  - `x_COAL`: Coal 🪨
  - `x_HYDRO`: Hydroelectric power 💧
  - `x_NUCLEAR`: Nuclear ⚛️
  - `x_SOLAR`: Solar power ☀️
  - `x_WINDPOW`: Wind power 🌪️
  - `x_LIGNITE`: Lignite 🔥
  - `x_CONSUMPTION`: Total electricity consumption ⚡
  - `x_RESIDUAL_LOAD`: Electricity consumed after renewable energy usage 🌱
  - `x_NET_IMPORT`: Electricity imported from Europe 🇪🇺
  - `x_NET_EXPORT`: Electricity exported to Europe 🌍
  - `DE_FR_EXCHANGE`: Electricity flow from Germany to France 🇩🇪➡️🇫🇷
  - `FR_DE_EXCHANGE`: Electricity flow from France to Germany 🇫🇷➡️🇩🇪

- **Y_train.csv**: Training output data
  - **2 columns**
  - `ID`: Unique identifier (same as input data)
  - `TARGET`: Daily variation in electricity futures price (24h maturity) 💹
